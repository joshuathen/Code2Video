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
        self.setup_layout("Scalar Multiplication: Stretching the Arrow", [
            "Scaling stretches or shrinks a vector.",
            "Multiplying by 3 triples the arrow length.",
            "Negative scalars reverse the vector direction."
        ])
        
        # Load Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        mag = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        
        # Initial State
        v_color = "#FFD700"
        v_start = ORIGIN
        v_end = RIGHT * 1.5 + UP * 0.5
        v_vec = Arrow(v_start, v_end, color=v_color, buff=0)
        v_label = MathTex(r"\\vec{v}", color=v_color).next_to(v_vec.get_end(), UP)
        vector_animation = VGroup(v_vec, v_label)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(v_color)
        self.place_in_area(vector_animation, 'D3', 'F6', scale_factor=0.8)
        self.place_at_grid(ruler, 'B3', scale_factor=0.5)
        self.play(Create(v_vec), Write(v_label), FadeIn(ruler))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(v_color)
        scalar_label = MathTex(r"3\\vec{v}", color=v_color)
        self.place_at_grid(scalar_label, 'B4', scale_factor=0.7)
        self.place_at_grid(mag, 'C4', scale_factor=0.5)
        
        v3 = Arrow(v_start, v_end * 3, color=v_color, buff=0)
        self.play(
            ReplacementTransform(v_vec, v3),
            ReplacementTransform(v_label, scalar_label),
            FadeIn(mag)
        )
        self.wait(2)
        
        # === Animation for Lecture Line 3 ===
        neg_color = "#FF4500"
        self.lecture[2].set_color(neg_color)
        
        neg_v = Arrow(v_start, -v_end, color=neg_color, buff=0)
        neg_label = MathTex(r"-\\vec{v}", color=neg_color).next_to(neg_v.get_end(), DOWN)
        
        # Use a group to satisfy the requirement of placing the grid group
        grid_group = VGroup(neg_v, neg_label)
        self.place_in_area(grid_group, 'B2', 'E5', scale_factor=0.6)
        
        self.play(
            FadeOut(ruler), FadeOut(mag),
            ReplacementTransform(v3, neg_v),
            ReplacementTransform(scalar_label, neg_label)
        )
        self.wait(2)
