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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Mathematics challenges our dimensional intuition.",
            "Infinity bridges the gap.",
            "A simple line becomes area."
        ]
        self.setup_layout("Conclusion: The Beauty of Limits", lecture_lines)
        
        # Elements
        dimension_label = Text("DIMENSION", color=BLUE)
        limit_label = Text("LIMIT", color=YELLOW)
        line_obj = Line(LEFT, RIGHT, color=RED)
        area_obj = Square(side_length=1.5, color=GREEN, fill_opacity=0.3)
        # Placeholder for the requested asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        # In a real environment, this would be: fractal_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        # Since it is a placeholder/none.svg, we use a simple Dot as a proxy
        fractal_icon = Dot(color=PURPLE)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(dimension_label, 'B2', scale_factor=0.7)
        self.place_at_grid(limit_label, 'B5', scale_factor=0.7)
        self.play(FadeIn(dimension_label), FadeIn(limit_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        bridge_arrow = Arrow(dimension_label.get_right(), limit_label.get_left(), color=WHITE)
        self.place_at_grid(fractal_icon, 'C3', scale_factor=2.0)
        self.play(Create(bridge_arrow), FadeIn(fractal_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.place_in_area(line_obj, 'C2', 'D2', scale_factor=0.8)
        self.place_in_area(area_obj, 'C5', 'D5', scale_factor=0.8)
        self.play(Create(line_obj), FadeIn(area_obj))
        self.play(Transform(line_obj, area_obj))
        self.wait(2)
