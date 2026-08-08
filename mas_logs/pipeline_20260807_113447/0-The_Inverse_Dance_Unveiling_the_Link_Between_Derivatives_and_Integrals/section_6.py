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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Application and Summary", [
            "Differentiation and integration are truly inverse processes.",
            "They are two sides of the same coin.",
            "The \"Inverse Dance\" is complete."
        ])
        
        # Define colors
        COLOR_SLOPE = YELLOW_A
        COLOR_AREA = BLUE_A
        COLOR_FTC = GREEN_A
        COLOR_SCALE = GRAY_B

        # === Animation for Lecture Line 1 ===
        # Animate a balance scale with a 'Slope' icon on one side and an 'Area' icon on the other.
        self.play(self.lecture[0].animate.set_color(COLOR_SLOPE))

        # Fulcrum
        fulcrum = Triangle(color=COLOR_SCALE).scale(0.3)
        self.place_at_grid(fulcrum, "E4")
        
        # Beam (starts tilted: left side up, right side down)
        beam = Line(LEFT, RIGHT, color=COLOR_SCALE).scale(2.2)
        beam.rotate(20 * DEGREES)
        self.place_at_grid(beam, "D4")

        # Slope icon (on the left tray, higher up due to tilt)
        slope_icon = MathTex("\\frac{dy}{dx}", color=COLOR_SLOPE)
        slope_label = Text("Slope", font_size=20, color=COLOR_SLOPE)
        slope_group = VGroup(slope_icon, slope_label).arrange(DOWN, buff=0.1)
        
        # Area icon (on the right tray, lower down due to tilt)
        area_icon = MathTex("\\int f(x) dx", color=COLOR_AREA)
        area_label = Text("Area", font_size=20, color=COLOR_AREA)
        area_group = VGroup(area_icon, area_label).arrange(DOWN, buff=0.1)

        # Positioning icons based on beam rotation to show initial imbalance
        # Using B3 (up) and E5 (down) to maintain a clear tilt before balance
        self.place_at_grid(slope_group, "B3", scale_factor=0.8)
        self.place_at_grid(area_group, "E5", scale_factor=0.7)

        self.play(FadeIn(fulcrum), Create(beam))
        self.play(FadeIn(slope_group), FadeIn(area_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Bring the scales to a perfect balance using the 'FTC' symbol as the fulcrum base.
        self.play(self.lecture[1].animate.set_color(COLOR_FTC))

        # FTC symbol placed at F4 (the base) to avoid overlapping the fulcrum's body (E4)
        ftc_symbol = MathTex("FTC", color=COLOR_FTC)
        self.place_at_grid(ftc_symbol, "F4", scale_factor=0.6)
        
        # Animate balance: rotate beam to horizontal and move icons to Row C to sit on the beam
        # Following critic instruction for final balanced positions at C3 and C5.
        self.play(
            beam.animate.rotate(-20 * DEGREES),
            slope_group.animate.move_to(self.grid["C3"]),
            area_group.animate.move_to(self.grid["C5"]),
            FadeIn(ftc_symbol),
            fulcrum.animate.set_color(COLOR_FTC),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display the summary text 'Differentiation and Integration are Inverses' in white (#FFFFFF).
        self.play(self.lecture[2].animate.set_color(WHITE))

        # Summary text placed in Row A (A2-A5) as per critic suggestion to avoid crowding
        summary_text = Text("Differentiation and Integration\nare Inverses", font_size=24, color=WHITE)
        self.place_in_area(summary_text, "A2", "A5", scale_factor=0.8)

        self.play(Write(summary_text))
        self.wait(2)

# Issue 42 resolved: ftc_symbol placed at F4 with scale 0.6 and not moved to E4 to prevent overlap.
# Issue 43 resolved: slope_group and area_group now balance at C3 and C5 (Row C) respectively.
# Issue 44 resolved: summary_text moved to a single-row area (A2-A5) at scale 0.8 to reduce crowding.
