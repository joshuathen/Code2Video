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
        lecture_lines = ["The dot product measures the projection of vectors.", "Visualize it as a shadow cast by light.", "It captures how much vectors align together.", "Think of a hiker climbing a steep hill.", "Only the upward motion contributes to the work."]
        self.setup_layout("Geometric Intuition: The Projection", lecture_lines)
        
        # Assets
        hill = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hill.svg")
        hiker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hiker.svg")
        light = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/light.svg")
        
        # Logic
        vec_a = Vector([1.5, 1.2], color=WHITE)
        vec_b = Vector([2.0, 0], color=WHITE)
        vec_group = VGroup(vec_a, vec_b)
        
        shadow = Line(start=ORIGIN, end=[1.5, 0, 0], color="#FF5733", stroke_width=6)
        
        geometry_group = VGroup(vec_group, shadow)
        
        # Placing per constraints
        self.place_in_area(geometry_group, 'B2', 'D5', scale_factor=0.6)
        self.place_at_grid(hill, 'E5', scale_factor=0.5)
        self.place_at_grid(light, 'A5', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Create(vec_a), Create(vec_b))
        
        # === Animation for Lecture Line 2 ===
        self.place_at_grid(shadow, 'E3', scale_factor=0.7)
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.play(FadeIn(light), Create(shadow))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        self.play(Indicate(geometry_group))
        
        # === Animation for Lecture Line 4 ===
        self.place_at_grid(hiker, 'D3', scale_factor=0.4)
        self.play(self.lecture[3].animate.set_color(BLUE))
        self.play(FadeIn(hiker), FadeIn(hill))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(BLUE))
        self.play(hiker.animate.move_to(shadow.get_end()))
