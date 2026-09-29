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
        lecture_lines = [
            "State-space maps to the Sierpinski Triangle.",
            "Recursive fractal geometry emerges from restricted moves.",
            "Triangles represent states; edges represent valid transitions."
        ]
        self.setup_layout("Mapping to the Sierpinski Triangle", lecture_lines)
        
        # Helper to create Sierpinski triangle
        def get_sierpinski(order, side_length=2.5):
            if order == 0:
                return Triangle(color=WHITE)
            
            sub = get_sierpinski(order - 1, side_length / 2)
            t1 = sub.copy().shift(UP * (side_length / (2 * np.sqrt(3))))
            t2 = sub.copy().shift(LEFT * (side_length / 2) + DOWN * (side_length / (2 * np.sqrt(3))))
            t3 = sub.copy().shift(RIGHT * (side_length / 2) + DOWN * (side_length / (2 * np.sqrt(3))))
            return VGroup(t1, t2, t3)

        # Assets
        # Note: icon/none.svg is a placeholder, as per instructions
        # Use simple shapes as proxies
        asset1 = Dot(color=WHITE).scale(0.5) 
        asset2 = Dot(color=GREEN).scale(0.5)

        # === Animation for Lecture Line 1 ===
        sierpinski_base = self.place_in_area(get_sierpinski(1), 'C2', 'E5', scale_factor=0.6)
        self.place_at_grid(asset1, 'B2')
        self.play(Create(sierpinski_base), FadeIn(asset1), run_time=2)
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        sierpinski_mid = self.place_in_area(get_sierpinski(2), 'C2', 'E5', scale_factor=0.65)
        self.play(ReplacementTransform(sierpinski_base, sierpinski_mid), run_time=2)
        self.play(self.lecture[1].animate.set_color("#FFD700"))

        # === Animation for Lecture Line 3 ===
        sierpinski_full = self.place_in_area(get_sierpinski(3), 'C2', 'E5', scale_factor=0.7)
        self.place_at_grid(asset2, 'E6')
        self.play(ReplacementTransform(sierpinski_mid, sierpinski_full), FadeIn(asset2), run_time=2)
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.wait(2)
