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
        self.setup_layout("The Transformation Matrix", [
            "A matrix stores new vector coordinates.",
            "Column one represents the new i.",
            "Column two represents the new j.",
            "Column three represents the new k."
        ])
        
        # Define matrix
        matrix = Matrix([[r"a", r"d", r"g"], [r"b", r"e", r"h"], [r"c", r"f", r"i"]])
        self.place_at_grid(matrix, 'B3', scale_factor=1.2)
        
        # Define indicators
        i_indicator = MathTex(r"\vec{i}", color="#00FFFF")
        j_indicator = MathTex(r"\vec{j}", color="#00FFFF")
        k_indicator = MathTex(r"\vec{k}", color="#00FFFF")
        
        self.place_at_grid(i_indicator, 'B2', scale_factor=0.9)
        self.place_at_grid(j_indicator, 'B3', scale_factor=0.9)
        self.place_at_grid(k_indicator, 'B4', scale_factor=0.9)
        
        # Hide initially
        i_indicator.set_opacity(0)
        j_indicator.set_opacity(0)
        k_indicator.set_opacity(0)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.play(Create(matrix))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        col1 = matrix.get_columns()[0]
        self.play(col1.animate.set_color("#00FFFF"), i_indicator.animate.set_opacity(1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        col2 = matrix.get_columns()[1]
        self.play(col2.animate.set_color("#00FFFF"), j_indicator.animate.set_opacity(1))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FFFF"))
        col3 = matrix.get_columns()[2]
        self.play(col3.animate.set_color("#00FFFF"), k_indicator.animate.set_opacity(1))
        self.wait(1)
