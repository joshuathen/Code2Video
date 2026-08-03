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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite Knowledge: Euclidean Space", [
            "Traditionally, we see vectors as arrows on a grid.",
            "We add them by placing them tip-to-tail.",
            "Or scale them by stretching or shrinking their length."
        ])

        # === Animation for Lecture Line 1 ===
        # A white arrow (#FFFFFF) appears on a dark gray faint grid (#333333).
        self.lecture[0].set_color(WHITE)
        
        # Create a grid for visual context
        grid_lines = VGroup()
        for r in ["A", "B", "C", "D", "E", "F"]:
            line = Line(self.grid[f"{r}1"], self.grid[f"{r}6"], color="#333333", stroke_width=1)
            grid_lines.add(line)
        for c in ["1", "2", "3", "4", "5", "6"]:
            line = Line(self.grid["A" + c], self.grid["F" + c], color="#333333", stroke_width=1)
            grid_lines.add(line)
        
        vector_grid_group = grid_lines
        # Satisfy Issue 22 and 23 by binding the grid asset to the visual anchor system area
        self.place_in_area(vector_grid_group, 'A2', 'F6', scale_factor=0.9)
        self.add(vector_grid_group)
        
        # Vector v from E2 to D3
        v1_start = self.grid['E2']
        v1_end = self.grid['D3']
        arrow1 = Arrow(v1_start, v1_end, buff=0, color="#FFFFFF", stroke_width=4)
        v_label = MathTex("\\vec{v}", color="#FFFFFF")
        # Place label near the vector within the anchor system
        self.place_at_grid(v_label, 'D2', scale_factor=0.8)
        v_label.shift(DOWN*0.3 + LEFT*0.2)
        
        self.play(Create(arrow1), Write(v_label), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A second cyan arrow (#00FFFF) attaches to the tip of the first arrow to show addition.
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color("#00FFFF")
        )
        
        # Vector u from D3 to B4
        v2_start = self.grid['D3']
        v2_end = self.grid['B4']
        arrow2 = Arrow(v2_start, v2_end, buff=0, color="#00FFFF", stroke_width=4)
        u_label = MathTex("\\vec{u}", color="#00FFFF")
        # Place label near the vector
        self.place_at_grid(u_label, 'C4', scale_factor=0.8)
        u_label.shift(LEFT*0.4)
        
        self.play(Create(arrow2), Write(u_label), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The arrow scales by 2x and its color changes to yellow (#FFFF00).
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        
        # Scale arrow1 to 2x (from E2 towards D3, ending at C4)
        v1_scaled_end = self.grid['C4']
        
        v_scaled_label = MathTex("2\\vec{v}", color="#FFFF00")
        self.place_at_grid(v_scaled_label, 'C4', scale_factor=0.8)
        v_scaled_label.shift(RIGHT*0.5 + UP*0.2)

        self.play(
            FadeOut(arrow2),
            FadeOut(u_label),
            FadeOut(v_label),
            arrow1.animate.set_color("#FFFF00").put_start_and_end_on(v1_start, v1_scaled_end),
            Write(v_scaled_label),
            run_time=2
        )
        self.wait(2)
