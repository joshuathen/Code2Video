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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup layout with lecture lines from storyboard
        lecture_lines = [
            "First, e^A is defined by a matrix power series.",
            "Second, diagonalization provides a fast way to compute it.",
            "Finally, it solves continuous state evolution in physical systems."
        ]
        self.setup_layout("Summary and Key Takeaways", lecture_lines)
        
        # Initial dimming of lecture lines to highlight them one by one
        self.lecture.set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        item1 = Text("1. Power Series Definition", color="#FFFFFF", font_size=28)
        self.place_in_area(item1, "B2", "B6", scale_factor=0.7)
        self.play(Write(item1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        item2 = Text("2. Diagonalization Shortcut", color="#FFFF00", font_size=28)
        self.place_in_area(item2, "C2", "C6", scale_factor=0.7)
        self.play(Write(item2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        
        # Visual: Point 3 text with asset
        item3_text = Text("3. Differential Equation Solution", color="#00FF00", font_size=28)
        # Load asset: [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/system.svg]
        system_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/system.svg")
        system_icon.set_color(GREEN)
        
        # Group them to be "alongside" as per storyboard
        item3_group = VGroup(item3_text, system_icon).arrange(RIGHT, buff=0.3)
        self.place_in_area(item3_group, "D2", "D6", scale_factor=0.7)
        
        # Side-by-side comparison of scalar world vs matrix world
        comp_scalar_lbl = Text("Scalar:", font_size=22, color=WHITE)
        comp_scalar_val = Text("e^at", font_size=22, color=WHITE)
        scalar_group = VGroup(comp_scalar_lbl, comp_scalar_val).arrange(RIGHT, buff=0.2)
        
        comp_matrix_lbl = Text("Matrix:", font_size=22, color=GREEN)
        comp_matrix_val = Text("e^At", font_size=22, color=GREEN)
        matrix_group = VGroup(comp_matrix_lbl, comp_matrix_val).arrange(RIGHT, buff=0.2)
        
        # Arrange scalar and matrix groups side-by-side
        comparison = VGroup(scalar_group, matrix_group).arrange(RIGHT, buff=0.8)
        
        # Resolved Issue 35: Move to Row E to reduce gap
        # Resolved Issue 36: Scale to 0.7 for consistency
        self.place_in_area(comparison, "E2", "E6", scale_factor=0.7)
        
        self.play(
            Write(item3_group),
            FadeIn(comparison)
        )
        self.wait(3)
