-- ============================================================
-- STALE. DO NOT BUILD A DATABASE FROM THIS FILE.
--
-- This is a pg_dump taken 2025-10-23 and never regenerated. It describes seven
-- of the live database's seventeen tables, and it describes at least one of
-- them wrongly. Compare its tile_registry against `\d tile_registry` on the
-- live database (hpl_kb, 2026-08-26):
--
--   here                          live
--   ----                          ----
--   id integer NOT NULL (PK)      (no id column)
--   hpc_id character varying(100) hpc_id integer -> FK hpc_dictionary(hpc_id)
--   samples character varying(100) samples character varying
--   -                             slide_tile character varying NOT NULL (PK)
--   -                             dataset_id text NOT NULL
--   -                             hpc_vote_margin, hpc_neighbor_distance,
--                                 hpc_assigned_at, hpc_reference
--
-- Since tile_registry.hpc_id is now an integer with a foreign key into
-- hpc_dictionary(hpc_id), hpc_dictionary.hpc_id cannot still be the
-- varchar(100) declared below either — so the cluster reference tables here
-- have drifted too, by an amount nobody has measured.
--
-- Use backend/migrate_all.sql instead. It runs backend/migrate_kb_base_tables.sql
-- (transcribed from `\d` against the live database) followed by every column
-- migration in dependency order, and is idempotent.
--
-- Kept, not deleted, for two reasons: it is the only record of the four
-- hpc_* cluster reference tables that exists in git at all, and the shape it
-- describes is what the oldest rows in the live database were created under.
-- ============================================================

--
-- PostgreSQL database dump
--

\restrict nTdURf6PBMNaEBJ30hsp8pR4cSegUPbd7Byglqh1WN5nOSEOAdX1xZ8aGkxBnjE

-- Dumped from database version 18.0 (Postgres.app)
-- Dumped by pg_dump version 18.0 (Postgres.app)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: hpc_dictionary; Type: TABLE; Schema: public; Owner: vaidehipandya
--

CREATE TABLE public.hpc_dictionary (
    id integer NOT NULL,
    hpc_id character varying(100) NOT NULL,
    malignant boolean NOT NULL,
    cluster_homogeneity numeric(4,2),
    inflammation text,
    necrosis text
);


ALTER TABLE public.hpc_dictionary OWNER TO vaidehipandya;

--
-- Name: hpc_dictionary_id_seq; Type: SEQUENCE; Schema: public; Owner: vaidehipandya
--

CREATE SEQUENCE public.hpc_dictionary_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.hpc_dictionary_id_seq OWNER TO vaidehipandya;

--
-- Name: hpc_dictionary_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: vaidehipandya
--

ALTER SEQUENCE public.hpc_dictionary_id_seq OWNED BY public.hpc_dictionary.id;


--
-- Name: hpc_malignant_details; Type: TABLE; Schema: public; Owner: vaidehipandya
--

CREATE TABLE public.hpc_malignant_details (
    id integer NOT NULL,
    hpc_id character varying(20) NOT NULL,
    predominant_pattern text,
    second_pattern text,
    stroma_epithelium_ratio text,
    stromal_cellularity text,
    other_comments text
);


ALTER TABLE public.hpc_malignant_details OWNER TO vaidehipandya;

--
-- Name: hpc_malignant_details_id_seq; Type: SEQUENCE; Schema: public; Owner: vaidehipandya
--

CREATE SEQUENCE public.hpc_malignant_details_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.hpc_malignant_details_id_seq OWNER TO vaidehipandya;

--
-- Name: hpc_malignant_details_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: vaidehipandya
--

ALTER SEQUENCE public.hpc_malignant_details_id_seq OWNED BY public.hpc_malignant_details.id;


--
-- Name: hpc_non_malignant_details; Type: TABLE; Schema: public; Owner: vaidehipandya
--

CREATE TABLE public.hpc_non_malignant_details (
    id integer NOT NULL,
    hpc_id character varying(20) NOT NULL,
    tiles_contain_mostly text,
    second_most_common_feature text,
    other_notable_features text
);


ALTER TABLE public.hpc_non_malignant_details OWNER TO vaidehipandya;

--
-- Name: hpc_non_malignant_details_id_seq; Type: SEQUENCE; Schema: public; Owner: vaidehipandya
--

CREATE SEQUENCE public.hpc_non_malignant_details_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.hpc_non_malignant_details_id_seq OWNER TO vaidehipandya;

--
-- Name: hpc_non_malignant_details_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: vaidehipandya
--

ALTER SEQUENCE public.hpc_non_malignant_details_id_seq OWNED BY public.hpc_non_malignant_details.id;


--
-- Name: hpc_survival_analysis; Type: TABLE; Schema: public; Owner: vaidehipandya
--

CREATE TABLE public.hpc_survival_analysis (
    id integer NOT NULL,
    hpc_id character varying(20),
    coef double precision,
    z double precision,
    log2_p double precision,
    se double precision,
    p double precision,
    expcoef double precision,
    secoef double precision,
    expcoef_lower_95 double precision,
    expcoef_upper_95 double precision,
    coef_lower_95 double precision,
    coef_upper_95 double precision
);


ALTER TABLE public.hpc_survival_analysis OWNER TO vaidehipandya;

--
-- Name: hpc_survival_analysis_id_seq; Type: SEQUENCE; Schema: public; Owner: vaidehipandya
--

CREATE SEQUENCE public.hpc_survival_analysis_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.hpc_survival_analysis_id_seq OWNER TO vaidehipandya;

--
-- Name: hpc_survival_analysis_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: vaidehipandya
--

ALTER SEQUENCE public.hpc_survival_analysis_id_seq OWNED BY public.hpc_survival_analysis.id;


--
-- Name: hpl_profile_proportion; Type: TABLE; Schema: public; Owner: vaidehipandya
--

CREATE TABLE public.hpl_profile_proportion (
    id integer NOT NULL,
    samples character varying(130),
    hpc_id character varying(20),
    proportion double precision,
    slides character varying(170)
);


ALTER TABLE public.hpl_profile_proportion OWNER TO vaidehipandya;

--
-- Name: hpl_profile_proportion_id_seq; Type: SEQUENCE; Schema: public; Owner: vaidehipandya
--

CREATE SEQUENCE public.hpl_profile_proportion_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.hpl_profile_proportion_id_seq OWNER TO vaidehipandya;

--
-- Name: hpl_profile_proportion_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: vaidehipandya
--

ALTER SEQUENCE public.hpl_profile_proportion_id_seq OWNED BY public.hpl_profile_proportion.id;


--
-- Name: hpl_profile_summary; Type: TABLE; Schema: public; Owner: vaidehipandya
--

CREATE TABLE public.hpl_profile_summary (
    id integer NOT NULL,
    samples character varying(160) NOT NULL,
    slides character varying(150),
    cancer_type text,
    total_tiles integer,
    dominant_hpc character varying(130)
);


ALTER TABLE public.hpl_profile_summary OWNER TO vaidehipandya;

--
-- Name: hpl_profile_summary_id_seq; Type: SEQUENCE; Schema: public; Owner: vaidehipandya
--

CREATE SEQUENCE public.hpl_profile_summary_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.hpl_profile_summary_id_seq OWNER TO vaidehipandya;

--
-- Name: hpl_profile_summary_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: vaidehipandya
--

ALTER SEQUENCE public.hpl_profile_summary_id_seq OWNED BY public.hpl_profile_summary.id;


--
-- Name: tile_registry; Type: TABLE; Schema: public; Owner: vaidehipandya
--

CREATE TABLE public.tile_registry (
    id integer NOT NULL,
    samples character varying(100) NOT NULL,
    slides character varying(125) NOT NULL,
    tiles character varying(125) NOT NULL,
    hpc_id character varying(100),
    image_index integer,
    h5_source_path character varying(150)
);


ALTER TABLE public.tile_registry OWNER TO vaidehipandya;

--
-- Name: tile_registry_id_seq; Type: SEQUENCE; Schema: public; Owner: vaidehipandya
--

CREATE SEQUENCE public.tile_registry_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.tile_registry_id_seq OWNER TO vaidehipandya;

--
-- Name: tile_registry_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: vaidehipandya
--

ALTER SEQUENCE public.tile_registry_id_seq OWNED BY public.tile_registry.id;


--
-- Name: hpc_dictionary id; Type: DEFAULT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpc_dictionary ALTER COLUMN id SET DEFAULT nextval('public.hpc_dictionary_id_seq'::regclass);


--
-- Name: hpc_malignant_details id; Type: DEFAULT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpc_malignant_details ALTER COLUMN id SET DEFAULT nextval('public.hpc_malignant_details_id_seq'::regclass);


--
-- Name: hpc_non_malignant_details id; Type: DEFAULT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpc_non_malignant_details ALTER COLUMN id SET DEFAULT nextval('public.hpc_non_malignant_details_id_seq'::regclass);


--
-- Name: hpc_survival_analysis id; Type: DEFAULT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpc_survival_analysis ALTER COLUMN id SET DEFAULT nextval('public.hpc_survival_analysis_id_seq'::regclass);


--
-- Name: hpl_profile_proportion id; Type: DEFAULT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpl_profile_proportion ALTER COLUMN id SET DEFAULT nextval('public.hpl_profile_proportion_id_seq'::regclass);


--
-- Name: hpl_profile_summary id; Type: DEFAULT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpl_profile_summary ALTER COLUMN id SET DEFAULT nextval('public.hpl_profile_summary_id_seq'::regclass);


--
-- Name: tile_registry id; Type: DEFAULT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.tile_registry ALTER COLUMN id SET DEFAULT nextval('public.tile_registry_id_seq'::regclass);


--
-- Name: hpc_dictionary hpc_dictionary_hpc_id_key; Type: CONSTRAINT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpc_dictionary
    ADD CONSTRAINT hpc_dictionary_hpc_id_key UNIQUE (hpc_id);


--
-- Name: hpc_dictionary hpc_dictionary_pkey; Type: CONSTRAINT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpc_dictionary
    ADD CONSTRAINT hpc_dictionary_pkey PRIMARY KEY (id);


--
-- Name: hpc_malignant_details hpc_malignant_details_hpc_id_key; Type: CONSTRAINT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpc_malignant_details
    ADD CONSTRAINT hpc_malignant_details_hpc_id_key UNIQUE (hpc_id);


--
-- Name: hpc_malignant_details hpc_malignant_details_pkey; Type: CONSTRAINT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpc_malignant_details
    ADD CONSTRAINT hpc_malignant_details_pkey PRIMARY KEY (id);


--
-- Name: hpc_non_malignant_details hpc_non_malignant_details_hpc_id_key; Type: CONSTRAINT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpc_non_malignant_details
    ADD CONSTRAINT hpc_non_malignant_details_hpc_id_key UNIQUE (hpc_id);


--
-- Name: hpc_non_malignant_details hpc_non_malignant_details_pkey; Type: CONSTRAINT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpc_non_malignant_details
    ADD CONSTRAINT hpc_non_malignant_details_pkey PRIMARY KEY (id);


--
-- Name: hpc_survival_analysis hpc_survival_analysis_pkey; Type: CONSTRAINT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpc_survival_analysis
    ADD CONSTRAINT hpc_survival_analysis_pkey PRIMARY KEY (id);


--
-- Name: hpl_profile_proportion hpl_profile_proportion_pkey; Type: CONSTRAINT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpl_profile_proportion
    ADD CONSTRAINT hpl_profile_proportion_pkey PRIMARY KEY (id);


--
-- Name: hpl_profile_summary hpl_profile_summary_pkey; Type: CONSTRAINT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpl_profile_summary
    ADD CONSTRAINT hpl_profile_summary_pkey PRIMARY KEY (id);


--
-- Name: hpl_profile_summary hpl_profile_summary_unique_sample_slide; Type: CONSTRAINT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpl_profile_summary
    ADD CONSTRAINT hpl_profile_summary_unique_sample_slide UNIQUE (samples, slides);


--
-- Name: tile_registry tile_registry_pkey; Type: CONSTRAINT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.tile_registry
    ADD CONSTRAINT tile_registry_pkey PRIMARY KEY (id);


--
-- Name: hpc_malignant_details hpc_malignant_details_hpc_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpc_malignant_details
    ADD CONSTRAINT hpc_malignant_details_hpc_id_fkey FOREIGN KEY (hpc_id) REFERENCES public.hpc_dictionary(hpc_id) ON DELETE CASCADE;


--
-- Name: hpc_non_malignant_details hpc_non_malignant_details_hpc_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpc_non_malignant_details
    ADD CONSTRAINT hpc_non_malignant_details_hpc_id_fkey FOREIGN KEY (hpc_id) REFERENCES public.hpc_dictionary(hpc_id) ON DELETE CASCADE;


--
-- Name: hpc_survival_analysis hpc_survival_analysis_hpc_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpc_survival_analysis
    ADD CONSTRAINT hpc_survival_analysis_hpc_id_fkey FOREIGN KEY (hpc_id) REFERENCES public.hpc_dictionary(hpc_id) ON DELETE CASCADE;


--
-- Name: hpl_profile_proportion hpl_profile_proportion_hpc_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpl_profile_proportion
    ADD CONSTRAINT hpl_profile_proportion_hpc_id_fkey FOREIGN KEY (hpc_id) REFERENCES public.hpc_dictionary(hpc_id) ON DELETE CASCADE;


--
-- Name: hpl_profile_proportion hpl_profile_proportion_sample_slide_fkey; Type: FK CONSTRAINT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.hpl_profile_proportion
    ADD CONSTRAINT hpl_profile_proportion_sample_slide_fkey FOREIGN KEY (samples, slides) REFERENCES public.hpl_profile_summary(samples, slides) ON DELETE CASCADE;


--
-- Name: tile_registry tile_registry_hpc_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: vaidehipandya
--

ALTER TABLE ONLY public.tile_registry
    ADD CONSTRAINT tile_registry_hpc_id_fkey FOREIGN KEY (hpc_id) REFERENCES public.hpc_dictionary(hpc_id) ON DELETE SET NULL;


--
-- PostgreSQL database dump complete
--

\unrestrict nTdURf6PBMNaEBJ30hsp8pR4cSegUPbd7Byglqh1WN5nOSEOAdX1xZ8aGkxBnjE

