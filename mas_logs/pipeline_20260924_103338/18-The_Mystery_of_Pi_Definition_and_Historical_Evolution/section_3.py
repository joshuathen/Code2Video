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
        self.setup_layout("Historical Evolution: The Method of Exhaustion", [
            "Archimedes squeezed the circle with polygons.", 
            "More sides get closer to the circle.", 
            "This trapped the value of Pi."
        ])
        
        # Define base objects
        circle = Circle(radius=1.5, color=BLUE)
        polygon = RegularPolygon(n=4, radius=1.5, color=RED)
        group = VGroup(circle, polygon)
        self.place_in_area(group, 'B3', 'E5', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        # Archimedes squeezed the circle with polygons.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(circle), Create(polygon))

        # === Animation for Lecture Line 2 ===
        # More sides get closer to the circle.
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        
        for n in [6, 12, 24, 48, 96]:
            new_polygon = RegularPolygon(n=n, radius=1.5, color=RED)
            # Maintain position
            new_polygon.move_to(polygon.get_center())
            self.play(Transform(polygon, new_polygon), run_time=0.5)

        # === Animation for Lecture Line 3 ===
        # This trapped the value of Pi.
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(GREEN))
        
        label = Text("Perimeter ≈ Circumference", font_size=20, color="#00FF00")
        self.place_in_area(label, 'D2', 'D4', scale_factor=0.7)
        self.play(Write(label))
        self.wait(2)
