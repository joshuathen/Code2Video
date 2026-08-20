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
            "Basis vectors i and j are building blocks.",
            "They span the 2D coordinate system.",
            "Any vector is their linear combination."
        ]
        self.setup_layout("Basis Vectors: The Building Blocks", lecture_lines)
        
        # Assets
        grid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        ruler_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        
        axes = Axes(x_range=[-1, 3], y_range=[-1, 3], axis_config={"include_tip": True}).scale(0.5)
        self.place_in_area(axes, 'B3', 'F6', scale_factor=0.6)
        
        i_vec = Vector([1, 0], color=YELLOW)
        j_vec = Vector([0, 1], color=BLUE)
        
        origin = axes.c2p(0, 0)
        i_vec.shift(origin - i_vec.get_start())
        j_vec.shift(origin - j_vec.get_start())
        
        i_label = MathTex(r"\\hat{i}", color=YELLOW).next_to(axes.c2p(1, 0), DR, buff=0.1)
        j_label = MathTex(r"\\hat{j}", color=BLUE).next_to(axes.c2p(0, 1), UP, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(grid_icon, "A2", scale_factor=0.3)
        self.play(FadeIn(grid_icon))
        self.play(Create(axes))
        self.play(GrowArrow(i_vec), Write(i_label), GrowArrow(j_vec), Write(j_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        grid_background = NumberPlane(x_range=[-1, 3], y_range=[-1, 3]).scale(0.5)
        self.place_in_area(grid_background, 'C3', 'E5', scale_factor=0.7)
        self.play(FadeIn(grid_background))
        self.play(self.lecture[0].animate.set_color(WHITE))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.place_at_grid(ruler_icon, "A5", scale_factor=0.3)
        self.play(FadeIn(ruler_icon))
        
        target_vec = Vector([2, 1], color=GREEN)
        target_vec.shift(origin - target_vec.get_start())
        
        vector_formula = MathTex(r"2\\hat{i} + 1\\hat{j}", color=GREEN)
        self.place_at_grid(vector_formula, 'E5', scale_factor=0.9)
        
        self.play(GrowArrow(target_vec), Write(vector_formula))
        self.wait(2)
