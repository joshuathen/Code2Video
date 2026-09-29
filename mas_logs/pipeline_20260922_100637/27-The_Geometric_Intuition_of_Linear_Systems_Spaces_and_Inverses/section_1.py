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
        lecture_lines = ["Linear systems map inputs to target vectors.", "Matrix columns define the new coordinate space.", "We search for x mapping to b."]
        self.setup_layout("Visualizing Linear Systems", lecture_lines)
        
        # Grid setup
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": True}).scale(0.5)
        plane = NumberPlane(x_range=[-3, 3], y_range=[-3, 3]).scale(0.5)
        
        basis_v1 = Vector([1, 0.5], color=YELLOW)
        basis_v2 = Vector([-0.5, 1], color=BLUE)
        
        # Asset integration
        plane_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg")
        vector_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vector.svg")
        
        basis_group = VGroup(plane, axes, basis_v1, basis_v2, plane_img, vector_icon)
        self.place_in_area(basis_group, 'A4', 'F6', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(plane), FadeIn(axes), FadeIn(plane_img))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.play(GrowArrow(basis_v1), GrowArrow(basis_v2), FadeIn(vector_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        target_b = Vector([0.5, 1.5], color=GREEN)
        self.place_at_grid(target_b, 'B5', scale_factor=1.2)
        label = MathTex('b', color=GREEN).scale(0.8).next_to(target_b.get_end(), RIGHT, buff=0.1)
        self.play(Create(target_b), Write(label))
        self.wait(2)
