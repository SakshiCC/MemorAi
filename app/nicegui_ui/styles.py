from __future__ import annotations

from nicegui import ui


# ============================================================
# MEMORAI VISUAL DESIGN SYSTEM
# ============================================================


def apply_memorai_styles() -> None:

    # ========================================================
    # FONTS
    #
    # DM Serif Display -> headings
    # Nunito Sans      -> UI/body
    # ========================================================

    ui.add_head_html(
        """
        <link rel="preconnect"
              href="https://fonts.googleapis.com">

        <link rel="preconnect"
              href="https://fonts.gstatic.com"
              crossorigin>

        <link
            href="https://fonts.googleapis.com/css2?
            family=DM+Serif+Display:ital@0;1&
            family=Nunito+Sans:
            opsz,wght@6..12,400;6..12,500;
            6..12,600;6..12,700&
            display=swap"
            rel="stylesheet"
        >
        """
        .replace("\n", "")
        .replace("            ", "")
    )

    # ========================================================
    # CSS
    # ========================================================

    ui.add_css(
        """
        /* ==================================================
           BASE TOKENS
        ================================================== */

        :root {

            --memorai-sidebar-width: 282px;

            --memorai-purple: #7b42d3;
            --memorai-purple-hover: #7038c9;
            --memorai-lavender: #b48af4;
            --memorai-pink: #db71c9;

            --memorai-green: #34a675;
            --memorai-red: #dc4d4d;
            --memorai-orange: #e5a23b;

            --memorai-radius-sm: 12px;
            --memorai-radius-md: 18px;
            --memorai-radius-lg: 27px;
            --memorai-radius-pill: 999px;

            --memorai-speed: 180ms;
        }


        /* ==================================================
           LIGHT THEME
        ================================================== */

        body.body--light {

            --m-page:
                #f8f4ff;

            --m-sidebar:
                #f6f0ff;

            --m-topbar:
                rgba(251, 248, 255, 0.92);

            --m-card:
                #fffdfd;

            --m-card-soft:
                #fbf8ff;

            --m-input:
                #fffdfd;

            --m-border:
                #e4ddea;

            --m-border-strong:
                #d8cfdf;

            --m-text:
                #21142b;

            --m-body:
                #4d4259;

            --m-muted:
                #796d86;

            --m-faint:
                #a399ad;

            --m-nav-active:
                #7c43d6;

            --m-nav-active-text:
                #ffffff;

            --m-purple-soft:
                #efe4ff;

            --m-success-soft:
                #def3e8;

            --m-shadow:
                0 12px 28px
                rgba(69, 48, 89, 0.09);

            --m-input-shadow:
                0 6px 18px
                rgba(69, 48, 89, 0.09);
        }


        /* ==================================================
           DARK THEME
        ================================================== */

        body.body--dark {

            --m-page:
                #130b1d;

            --m-sidebar:
                #191024;

            --m-topbar:
                rgba(24, 15, 35, 0.95);

            --m-card:
                #241a30;

            --m-card-soft:
                #281e35;

            --m-input:
                #241a30;

            --m-border:
                #41364c;

            --m-border-strong:
                #50425d;

            --m-text:
                #fff8ff;

            --m-body:
                #c9bbd4;

            --m-muted:
                #aa9ab8;

            --m-faint:
                #887a94;

            --m-nav-active:
                #b187f5;

            --m-nav-active-text:
                #170d22;

            --m-purple-soft:
                #38274b;

            --m-success-soft:
                #18372c;

            --m-shadow:
                none;

            --m-input-shadow:
                none;
        }


        /* ==================================================
           ROOT
        ================================================== */

        html,
        body,
        #q-app {

            min-height: 100%;

            margin: 0;

            background:
                var(--m-page);

            color:
                var(--m-text);

            font-family:
                "Nunito Sans",
                Arial,
                sans-serif;

            transition:
                background-color
                var(--memorai-speed)
                ease,
                color
                var(--memorai-speed)
                ease;
        }


        body {

            font-size: 16px;

            letter-spacing: -0.01em;
        }


        /* ==================================================
           TYPOGRAPHY
        ================================================== */

        .m-display,
        .m-page-title,
        .m-heading,
        .m-section-title {

            font-family:
                "DM Serif Display",
                Georgia,
                serif;

            color:
                var(--m-text);

            font-weight: 400;

            letter-spacing: -0.015em;
        }


        .m-display {

            font-size: 50px;

            line-height: 1.05;
        }


        .m-page-title {

            font-size: 40px;

            line-height: 1.12;
        }


        .m-heading {

            font-size: 30px;

            line-height: 1.2;
        }


        .m-section-title {

            font-size: 25px;

            line-height: 1.25;
        }


        .m-subtitle {

            color:
                var(--m-body);

            font-size: 18px;
        }


        .m-description {

            color:
                var(--m-muted);

            font-size: 15px;

            line-height: 1.6;
        }


        /* ==================================================
           PAGE STRUCTURE
        ================================================== */

        .m-page {

            width: 100%;

            min-height: 100vh;

            gap: 0 !important;

            align-items: stretch !important;

            background:
                var(--m-page);
        }


        .m-content {

            flex: 1;

            min-width: 0;

            padding:
                38px
                46px
                70px
                46px;
        }


        .m-content-inner {

            width: 100%;

            max-width: 940px;

            margin-left: auto;

            margin-right: auto;
        }


        /* ==================================================
           SIDEBAR
        ================================================== */

        .m-sidebar {

            width:
                var(--memorai-sidebar-width);

            min-width:
                var(--memorai-sidebar-width);

            height: 100vh;

            padding:
                22px
                16px;

            position: sticky;

            top: 0;

            background:
                var(--m-sidebar);

            border-right:
                1px solid
                var(--m-border);

            gap: 6px !important;
        }


        .m-logo-circle {

            width: 46px;

            height: 46px;

            border-radius: 50%;

            display: flex;

            align-items: center;

            justify-content: center;

            background:
                var(--memorai-purple);

            color: #fff;

            font-size: 23px;
        }


        .m-brand {

            font-family:
                "DM Serif Display",
                Georgia,
                serif;

            font-size: 25px;

            color:
                var(--m-text);
        }


        .m-brand-subtitle {

            font-size: 14px;

            color:
                var(--m-muted);
        }


        .m-nav {

            width: 100%;

            height: 48px;

            padding:
                0
                17px;

            border-radius:
                16px;

            justify-content:
                flex-start !important;

            color:
                var(--m-text) !important;

            background:
                transparent !important;

            font-size: 16px;

            font-weight: 600;

            text-transform: none;

            box-shadow: none !important;

            transition:
                background
                var(--memorai-speed)
                ease;
        }


        .m-nav:hover {

            background:
                var(--m-purple-soft)
                !important;
        }


        .m-nav-active {

            background:
                var(--m-nav-active)
                !important;

            color:
                var(--m-nav-active-text)
                !important;

            box-shadow:
                0 9px 22px
                rgba(124, 67, 214, 0.16)
                !important;
        }


        .m-sidebar-footer {

            color:
                var(--m-faint);

            font-size: 12px;
        }


        /* ==================================================
           TOPBAR
        ================================================== */

        .m-topbar {

            height: 66px;

            width: 100%;

            padding:
                0
                32px;

            background:
                var(--m-topbar);

            border-bottom:
                1px solid
                var(--m-border);

            align-items: center;

            justify-content: flex-end;

            gap: 9px !important;
        }


        .m-circle-button {

            width: 43px;

            height: 43px;

            border-radius: 50%;

            background:
                transparent !important;

            border:
                1px solid
                var(--m-border)
                !important;

            color:
                var(--m-text)
                !important;

            box-shadow: none !important;
        }


        .m-avatar {

            width: 43px;

            height: 43px;

            border-radius: 50%;

            display: flex;

            align-items: center;

            justify-content: center;

            background:
                var(--m-purple-soft);

            color:
                var(--m-text);

            font-weight: 700;
        }


        /* ==================================================
           CARDS
        ================================================== */

        .m-card {

            width: 100%;

            padding: 24px;

            border-radius:
                var(--memorai-radius-lg);

            background:
                var(--m-card);

            border:
                1px solid
                var(--m-border);

            box-shadow:
                var(--m-shadow);
        }


        .m-soft-card {

            width: 100%;

            padding: 18px;

            border-radius:
                22px;

            background:
                var(--m-card-soft);

            border:
                1px solid
                var(--m-border);
        }


        /* ==================================================
           INPUT
        ================================================== */

        .m-pill-input {

            width: 100%;
        }


        .m-pill-input
        .q-field__control {

            min-height: 66px !important;

            padding:
                0
                20px;

            border-radius:
                34px !important;

            background:
                var(--m-input);

            box-shadow:
                var(--m-input-shadow);

            color:
                var(--m-text);
        }


        .m-pill-input
        .q-field__control:before {

            border:
                1px solid
                var(--m-border)
                !important;

            border-radius:
                34px !important;
        }


        .m-pill-input
        .q-field__control:after {

            border:
                1.5px solid
                var(--memorai-purple)
                !important;

            border-radius:
                34px !important;
        }


        .m-pill-input
        input {

            color:
                var(--m-text);

            font-family:
                "Nunito Sans",
                sans-serif;

            font-size: 17px;
        }


        .m-pill-input
        input::placeholder {

            color:
                var(--m-muted);

            opacity: 1;
        }


        /* ==================================================
           BUTTONS
        ================================================== */

        .m-primary-button {

            min-height: 56px;

            padding:
                0
                26px;

            border-radius:
                28px;

            background:
                var(--memorai-purple)
                !important;

            color:
                #fff !important;

            font-size: 16px;

            font-weight: 700;

            text-transform: none;

            box-shadow:
                0 10px 24px
                rgba(124, 67, 214, 0.18)
                !important;
        }


        .m-primary-button:hover {

            background:
                var(--memorai-purple-hover)
                !important;
        }


        .m-round-purple-button {

            width: 49px;

            height: 49px;

            min-width: 49px;

            border-radius: 50%;

            background:
                var(--memorai-purple)
                !important;

            color:
                white !important;
        }


        .m-round-soft-button {

            width: 49px;

            height: 49px;

            min-width: 49px;

            border-radius: 50%;

            background:
                var(--m-purple-soft)
                !important;

            color:
                var(--m-text)
                !important;
        }


        .m-outline-button {

            min-height: 44px;

            padding:
                0
                20px;

            border-radius:
                var(--memorai-radius-pill);

            background:
                transparent !important;

            border:
                1px solid
                var(--m-border)
                !important;

            color:
                var(--m-text)
                !important;

            box-shadow: none !important;

            text-transform: none;
        }


        .m-filter-active {

            background:
                var(--m-nav-active)
                !important;

            border-color:
                var(--m-nav-active)
                !important;

            color:
                var(--m-nav-active-text)
                !important;
        }


        /* ==================================================
           CAMERA
        ================================================== */

        .m-video-frame {

            width: 100%;

            padding: 11px;

            border-radius:
                26px;

            background:
                var(--m-card);

            border:
                1px solid
                var(--m-border);

            box-shadow:
                var(--m-shadow);
        }


        .m-video {

            display: block;

            width: 100%;

            max-height: 580px;

            object-fit: contain;

            border-radius: 19px;

            background: #000;
        }


        /* ==================================================
           STATUS
        ================================================== */

        .m-status-dot {

            display: inline-block;

            width: 10px;

            height: 10px;

            border-radius: 50%;

            background:
                var(--memorai-green);

            box-shadow:
                0 0 0 4px
                rgba(52, 166, 117, 0.15);
        }


        .m-status-text {

            color:
                var(--m-text);

            font-weight: 700;

            font-size: 16px;
        }


        .m-status-muted {

            color:
                var(--m-muted);

            font-size: 14px;
        }


        /* ==================================================
           MEMORY ROWS
        ================================================== */

        .m-memory-row {

            width: 100%;

            padding:
                18px
                22px;

            border-radius:
                24px;

            background:
                var(--m-card);

            border:
                1px solid
                var(--m-border);

            box-shadow:
                var(--m-shadow);
        }


        .m-memory-name {

            color:
                var(--m-text);

            font-size: 17px;

            font-weight: 700;
        }


        .m-memory-description {

            color:
                var(--m-muted);

            font-size: 15px;
        }


        .m-memory-time {

            color:
                var(--m-muted);

            font-size: 14px;
        }


        /* ==================================================
           ANSWER
        ================================================== */

        .m-answer {

            width: 100%;

            padding:
                18px
                20px;

            margin-top: 8px;

            border-radius: 20px;

            background:
                var(--m-purple-soft);

            color:
                var(--m-text);

            font-size: 16px;

            line-height: 1.6;
        }


        /* ==================================================
           QUICK ACTION
        ================================================== */

        .m-quick {

            min-height: 48px;

            border-radius:
                24px;

            border:
                1px solid
                var(--m-border)
                !important;

            background:
                var(--m-card)
                !important;

            color:
                var(--m-text)
                !important;

            text-transform: none;

            box-shadow: none !important;
        }


        /* ==================================================
           SETTINGS
        ================================================== */

        .m-setting-row {

            width: 100%;

            min-height: 54px;

            align-items: center;

            justify-content:
                space-between;
        }


        .m-setting-label {

            font-size: 17px;

            color:
                var(--m-text);
        }


        /* ==================================================
           RESPONSIVE
        ================================================== */

        @media (max-width: 1000px) {

            :root {

                --memorai-sidebar-width:
                    225px;
            }


            .m-content {

                padding:
                    30px
                    25px
                    60px;
            }
        }


        @media (max-width: 760px) {

            .m-sidebar {

                display: none;
            }


            .m-content {

                padding:
                    24px
                    16px
                    50px;
            }


            .m-display {

                font-size: 38px;
            }
        }
        """
    )