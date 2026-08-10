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
            "Basis vectors 'i' and 'j' define the space.",
            "Any 2D vector is a combination of them.",
            "This grid movement is a linear combination."
        ]
        self.setup_layout("Linear Combinations and Basis Vectors", lecture_lines)
        
        # Grid/Axes
        axes = Axes(x_range=[-1, 4, 1], y_range=[-1, 4, 1], axis_config={"include_numbers": False}).scale(0.5)
        # Fix: Line 56 - Scale axes to 0.9 for better fit
        self.place_in_area(axes, 'B2', 'E5', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        i_vec = Vector(RIGHT, color="#FFD700")
        j_vec = Vector(UP, color="#32CD32")
        i_label = MathTex(r"\hat{i}", color="#FFD700")
        j_label = MathTex(r"\hat{j}", color="#32CD32")
        
        # Fix: Line 67, 69 - Better label positioning
        self.place_at_grid(i_vec, 'D4')
        self.place_at_grid(i_label, 'D5', scale_factor=0.7)
        self.place_at_grid(j_vec, 'D4')
        self.place_at_grid(j_label, 'C4', scale_factor=0.7)
        
        self.play(Create(axes), GrowArrow(i_vec), Write(i_label), GrowArrow(j_vec), Write(j_label))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        # Asset: Load and display grid asset for linear combination illustration
        grid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_in_area(grid_icon, 'C3', 'D4', scale_factor=0.5)
        
        # Visualizing 3*i + 2*j
        v1 = Vector(RIGHT, color="#FFD700").shift(axes.c2p(0, 0))
        v2 = Vector(RIGHT, color="#FFD700").shift(axes.c2p(1, 0))
        v3 = Vector(RIGHT, color="#FFD700").shift(axes.c2p(2, 0))
        h1 = Vector(UP, color="#32CD32").shift(axes.c2p(3, 0))
        h2 = Vector(UP, color="#32CD32").shift(axes.c2p(3, 1))
        
        vec_v = Vector(3*RIGHT + 2*UP, color=WHITE)
        vec_v.shift(axes.c2p(0,0))
        
        self.play(
            FadeIn(grid_icon),
            Create(v1), Create(v2), Create(v3),
            Create(h1), Create(h2),
            Create(vec_v)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF69B4")
        span_text = Text("Span", color=WHITE, font_size=30)
        # Fix: Line 96 - Better position for Span text
        self.place_at_grid(span_text, 'B4', scale_factor=1.0)
        self.play(FadeOut(axes), FadeOut(v1), FadeOut(v2), FadeOut(v3), FadeOut(h1), FadeOut(h2), FadeOut(vec_v), FadeOut(grid_icon), FadeIn(span_text))
        self.wait(2)
