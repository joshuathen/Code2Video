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
        lecture_lines = ["Basis vectors i and j are building blocks.", "Any 2D vector is their linear combination.", "They define the coordinate grid."]
        self.setup_layout("Basis Vectors: The Building Blocks", lecture_lines)
        
        # Grid Background - Switched from ImageMobject to SVGMobject as it is an SVG file
        grid_bg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_in_area(grid_bg, "B3", "F6", scale_factor=1.5)
        self.add(grid_bg)
        
        axes = Axes(x_range=[-1, 3], y_range=[-1, 3], axis_config={"include_numbers": False}).scale(0.6)
        self.place_in_area(axes, "B3", "F6", scale_factor=0.6)
        self.add(axes)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        i_hat = Vector(RIGHT, color="#00FFFF")
        j_hat = Vector(UP, color="#FFD700")
        
        # Placing at grid coordinates using axes.c2p
        i_vec = axes.c2p(1, 0)
        j_vec = axes.c2p(0, 1)
        origin = axes.c2p(0, 0)
        
        i_hat.shift(origin - i_hat.get_start())
        j_hat.shift(origin - j_hat.get_start())
        
        self.play(Create(i_hat))
        self.play(Create(j_hat))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF4500")
        
        v_label = MathTex(r"\\vec{v} = 3\\hat{i} + 2\\hat{j}", color="#FF4500")
        self.place_at_grid(v_label, "C4", scale_factor=0.75)
        
        v_vector = Vector(axes.c2p(3, 2) - origin, color="#FF4500")
        v_vector.shift(origin - v_vector.get_start())
        
        self.play(Create(v_vector))
        self.play(Write(v_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        grid_lines = NumberPlane(x_range=[-1, 3], y_range=[-1, 3]).scale(0.6).move_to(axes.get_center())
        self.play(FadeIn(grid_lines))
        self.wait(2)
