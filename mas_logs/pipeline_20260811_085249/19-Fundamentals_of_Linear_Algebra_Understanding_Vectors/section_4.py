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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Basis vectors i-hat and j-hat span the space.",
            "Every vector is a scaled combination of these.",
            "Grid coordinates map directly to these scales."
        ]
        self.setup_layout("Linear Combinations and Basis Vectors", lecture_lines)
        
        # Elements
        axes = Axes(x_range=[-1, 4, 1], y_range=[-1, 4, 1], axis_config={"include_tip": True})
        i_hat = Vector(RIGHT, color=YELLOW)
        j_hat = Vector(UP, color=BLUE)
        i_label = MathTex(r"\hat{i}", color=YELLOW).next_to(i_hat.get_end(), DOWN)
        j_label = MathTex(r"\hat{j}", color=BLUE).next_to(j_hat.get_end(), LEFT)
        
        vector_v = Vector(3*RIGHT + 2*UP, color=GREEN)
        v_label = MathTex(r"\vec{v} = 3\hat{i} + 2\hat{j}", color=GREEN)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_in_area(axes, 'B3', 'F6', scale_factor=0.7)
        self.play(Create(axes))
        i_hat.next_to(axes.c2p(0,0), RIGHT, buff=0)
        j_hat.next_to(axes.c2p(0,0), UP, buff=0)
        self.play(GrowArrow(i_hat), Write(i_label), GrowArrow(j_hat), Write(j_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        vector_v.shift(axes.c2p(0,0))
        self.play(GrowArrow(vector_v))
        self.place_at_grid(v_label, 'D4', scale_factor=0.9)
        self.play(Write(v_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        grid_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_in_area(grid_svg, 'B3', 'F6', scale_factor=0.3)
        self.play(FadeIn(grid_svg), run_time=2)
        self.wait(2)
