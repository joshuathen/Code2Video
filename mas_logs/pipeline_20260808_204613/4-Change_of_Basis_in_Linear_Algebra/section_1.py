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
        self.setup_layout("Prerequisites: The Language of Vectors", 
                          ["A basis defines our coordinate system.", 
                           "Vectors are displacements, not just numbers.", 
                           "Coordinates depend on the basis chosen."])
        
        # Assets
        grid_bg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        ruler_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        
        # Define base objects
        axes = Axes(x_range=[-2, 3], y_range=[-2, 3], axis_config={"include_tip": True}).scale(0.6)
        i_vec = Arrow(start=ORIGIN, end=axes.c2p(1, 0), color=WHITE, buff=0)
        j_vec = Arrow(start=ORIGIN, end=axes.c2p(0, 1), color=WHITE, buff=0)
        i_label = MathTex("i", color=WHITE).next_to(i_vec.get_end(), RIGHT, buff=0.1)
        j_label = MathTex("j", color=WHITE).next_to(j_vec.get_end(), UP, buff=0.1)
        basis = VGroup(axes, i_vec, j_vec, i_label, j_label)
        vector_units = VGroup(i_vec, j_vec)

        v_vec = Arrow(start=ORIGIN, end=axes.c2p(2, 1.5), color=YELLOW, buff=0)
        v_label = MathTex("v", color=YELLOW).next_to(v_vec.get_end(), UP+RIGHT, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.place_in_area(grid_bg, 'C3', 'E6', scale_factor=0.9)
        self.place_in_area(basis, 'C3', 'E6', scale_factor=0.9)
        self.place_at_grid(vector_units, 'C3', scale_factor=1.0)
        self.add(grid_bg)
        self.play(Create(basis), run_time=1.5)
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.play(GrowArrow(v_vec), Write(v_label))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        coord_label = MathTex("(2, 1.5)", color=YELLOW)
        self.place_at_grid(coord_label, 'E5', scale_factor=0.7)
        self.place_at_grid(ruler_asset, 'D3', scale_factor=0.5)
        self.play(Write(coord_label), FadeIn(ruler_asset))
        self.lecture[2].set_color(GREEN)
        self.wait(1)
