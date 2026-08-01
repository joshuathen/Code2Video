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

class Section3Scene(TeachingScene):
    def construct(self):
        # Initialize Scene with storyboard content
        title = "Building the Triangle of Power"
        lecture_lines = [
            "Meet the Triangle of Power, a unified notation.",
            "Place the Base at the bottom-left vertex.",
            "Put the Exponent at the top vertex.",
            "Set the Result at the bottom-right vertex.",
            "One geometric shape now holds all three parts."
        ]
        self.setup_layout(title, lecture_lines)
        
        # Colors
        BLUE_BASE = "#00CCFF"
        ORANGE_EXP = "#FF9900"
        PURPLE_RES = "#CC00FF"
        WHITE_TRI = "#FFFFFF"
        GOLD_GLOW = "#FFD700"

        # === Animation for Lecture Line 1 ===
        # Draw a white (#FFFFFF) triangle outline in the center.
        self.lecture[0].set_color(WHITE_TRI)
        
        # We anchor the triangle to the grid by using specific grid coordinates for its vertices.
        # This addresses Issue 29 by ensuring the shape is geometrically linked to the grid system.
        # Vertices: Top (B4), Bottom-Left (E3), Bottom-Right (E5).
        triangle = Polygon(
            self.grid["B4"], self.grid["E3"], self.grid["E5"],
            color=WHITE_TRI, stroke_width=4
        )
        
        self.play(Create(triangle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Place the blue (#00CCFF) number '2' at the bottom-left vertex.
        self.lecture[1].set_color(BLUE_BASE)
        base_2 = MathTex("2", color=BLUE_BASE)
        # Precise grid anchoring at 'E3' as per Issue 30.
        self.place_at_grid(base_2, 'E3', scale_factor=0.8)
        self.play(Write(base_2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Put the orange (#FF9900) number '3' at the top vertex.
        self.lecture[2].set_color(ORANGE_EXP)
        exponent_3 = MathTex("3", color=ORANGE_EXP)
        # Precise grid anchoring at 'B4' as per Issue 31.
        self.place_at_grid(exponent_3, 'B4', scale_factor=0.8)
        self.play(Write(exponent_3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Set the Result at the bottom-right vertex.
        self.lecture[3].set_color(PURPLE_RES)
        result_8 = MathTex("8", color=PURPLE_RES)
        # Precise grid anchoring at 'E5' as per Issue 31.
        self.place_at_grid(result_8, 'E5', scale_factor=0.8)
        self.play(Write(result_8))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The entire triangle structure glows with a golden (#FFD700) light.
        self.lecture[4].set_color(GOLD_GLOW)
        
        # Create a glow effect by layering a slightly larger, translucent copy.
        glow_triangle = triangle.copy().set_stroke(GOLD_GLOW, width=12).set_opacity(0.3)
        
        self.play(
            triangle.animate.set_color(GOLD_GLOW),
            FadeIn(glow_triangle),
            rate_func=there_and_back,
            run_time=2
        )
        
        # Permanent state change for the end of the section
        self.play(
            triangle.animate.set_color(GOLD_GLOW).set_stroke(width=6),
            glow_triangle.animate.set_opacity(0.5).set_stroke(width=10),
            base_2.animate.scale(1.1),
            exponent_3.animate.scale(1.1),
            result_8.animate.scale(1.1),
            run_time=1
        )
        self.wait(2)
