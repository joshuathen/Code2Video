from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section1Scene(TeachingScene):
    def construct(self):
        # Data from shared state
        title_text = "Introduction: The Privacy Paradox"
        lecture_lines = [
            "Digital contact tracing helps stop virus spread quickly.",
            "Centralized systems risk creating a mass surveillance state.",
            "DP-3T ensures notification without revealing individual identities."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors from storyboard
        COLOR_ALICE = "#58D68D"
        COLOR_BOB = "#5DADE2"
        COLOR_VIRUS = "#E74C3C"
        COLOR_EYE = WHITE
        COLOR_X = "#C0392B"
        COLOR_PHONE = WHITE
        COLOR_PRIVACY = "#F1C40F"
        
        # Assets
        PHONE_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg"

        # === Animation for Lecture Line 1 ===
        # Highlights: Digital contact tracing & virus spread
        self.play(self.lecture[0].animate.set_color(COLOR_VIRUS), run_time=0.5)

        # Characters Alice and Bob
        alice = Circle(radius=0.35, color=COLOR_ALICE, fill_opacity=0.6)
        self.place_at_grid(alice, "B2")
        alice_label = Text("Alice", font_size=18, color=COLOR_ALICE)
        self.place_at_grid(alice_label, "A2")

        bob = Circle(radius=0.35, color=COLOR_BOB, fill_opacity=0.6)
        self.place_at_grid(bob, "B5")
        bob_label = Text("Bob", font_size=18, color=COLOR_BOB)
        self.place_at_grid(bob_label, "A5")

        # Virus icon appearing between them
        virus_core = Circle(radius=0.2, color=COLOR_VIRUS, fill_opacity=1)
        spikes = VGroup(*[
            Line(ORIGIN, 0.3 * RIGHT).rotate(angle).set_color(COLOR_VIRUS)
            for angle in np.linspace(0, TAU, 12, endpoint=False)
        ]).move_to(virus_core)
        virus = VGroup(virus_core, spikes)
        self.place_in_area(virus, "B3", "B4")

        self.play(
            FadeIn(alice), FadeIn(alice_label),
            FadeIn(bob), FadeIn(bob_label),
            run_time=1
        )
        self.play(GrowFromCenter(virus), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlights: Mass surveillance risk
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_X),
            run_time=0.5
        )

        # Big Brother Eye icon
        eye_white = Ellipse(width=1.6, height=0.8, color=COLOR_EYE, fill_opacity=1)
        eye_iris = Circle(radius=0.3, color=BLUE_E, fill_opacity=1)
        eye_pupil = Circle(radius=0.1, color=BLACK, fill_opacity=1)
        eye = VGroup(eye_white, eye_iris, eye_pupil)
        # Fix Issue 51: Move eye to D3-E4
        self.place_in_area(eye, "D3", "E4")

        # Large red 'X' canceling the eye
        cancel_x = VGroup(
            Line(UL, DR, color=COLOR_X, stroke_width=15),
            Line(DL, UR, color=COLOR_X, stroke_width=15)
        ).scale(1.2)
        # Fix Issue 51: Move cancel_x to D3-E4
        self.place_in_area(cancel_x, "D3", "E4")

        self.play(FadeIn(eye), run_time=1)
        self.play(Create(cancel_x), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlights: DP-3T privacy guarantee
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_PRIVACY),
            run_time=0.5
        )

        # Fix Issue 30: Use PHONE_ASSET (SVG)
        # Fix Issue 52: Move phones to C2 and C5
        phone_a = SVGMobject(PHONE_ASSET, color=COLOR_PHONE, fill_opacity=0.1)
        self.place_at_grid(phone_a, "C2", scale_factor=0.6)
        
        phone_b = SVGMobject(PHONE_ASSET, color=COLOR_PHONE, fill_opacity=0.1)
        self.place_at_grid(phone_b, "C5", scale_factor=0.6)

        # 'Privacy Preserved' text
        # Fix Issue 53: Move privacy_text to D2-D5
        privacy_text = Text("Privacy Preserved", font_size=32, color=COLOR_PRIVACY)
        self.place_in_area(privacy_text, "D2", "D5")

        # Simple glow effects around phones
        glow_a = Circle(radius=0.5, color=COLOR_PRIVACY, fill_opacity=0.2).move_to(phone_a)
        glow_b = Circle(radius=0.5, color=COLOR_PRIVACY, fill_opacity=0.2).move_to(phone_b)

        # Clear surveillance icons and show privacy solution
        self.play(
            FadeOut(eye), FadeOut(cancel_x), FadeOut(virus),
            run_time=1
        )
        self.play(
            FadeIn(phone_a), FadeIn(phone_b),
            FadeIn(glow_a), FadeIn(glow_b),
            run_time=1
        )
        self.play(Write(privacy_text), run_time=1)
        self.wait(3)
