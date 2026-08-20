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
        self.setup_layout("Linear Combinations and Span", [
            "Linear combination combines scaled vectors.",
            "Span is the set of all reachable points.",
            "Span creates a 'coverage area' on a grid."
        ])

        # Colors
        VEC_COLOR = "#33FF57"
        RESULT_COLOR = "#FF33FF"

        # Assets
        grid_bg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_in_area(grid_bg, 'B2', 'E5', scale_factor=1.2)
        self.add(grid_bg)

        # Setup Basis
        i_hat = Vector([1, 0], color=VEC_COLOR)
        j_hat = Vector([0, 1], color=VEC_COLOR)
        
        self.place_at_grid(i_hat, 'D3', scale_factor=0.8)
        self.place_at_grid(j_hat, 'D4', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(VEC_COLOR)
        self.play(Create(i_hat), Create(j_hat))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(VEC_COLOR)
        v_scaled = Vector([0.5, 0.7], color=VEC_COLOR)
        self.place_at_grid(v_scaled, 'E3', scale_factor=1.0)
        self.play(Create(v_scaled))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RESULT_COLOR)
        res_vec = Vector([0.5, 1.7], color=RESULT_COLOR)
        self.place_at_grid(res_vec, 'E4', scale_factor=1.0)
        
        # Group for area placement requirement
        vector_group = VGroup(i_hat, j_hat, v_scaled, res_vec)
        self.place_in_area(vector_group, 'C2', 'E5', scale_factor=0.9)
        
        self.play(Create(res_vec))
        self.wait(2)
