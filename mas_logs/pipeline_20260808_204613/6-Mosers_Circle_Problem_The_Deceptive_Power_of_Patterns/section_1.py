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
        self.setup_layout("The Setup: Connecting Points on a Circle", [
            "Place points on a circle.", 
            "Connect every pair with chords.", 
            "Count the resulting regions.", 
            "One point gives one region.", 
            "Two points give two regions."
        ])
        
        # Using asset (Note: Ensure the file path exists or fallback to a standard Mobject)
        circle = Circle(radius=1.0, color=WHITE)
        self.place_at_grid(circle, 'B5', scale_factor=0.6)
        self.add(circle)
        
        # Pre-create count label to avoid flicker
        count_label = Text("Regions: 0", font_size=24, color=WHITE)
        self.place_at_grid(count_label, 'E5', scale_factor=0.7)
        self.add(count_label)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF5733")
        points = [Dot(circle.point_from_proportion(i/4), color=YELLOW) for i in range(4)]
        self.play(FadeIn(points[0]), FadeIn(points[1]))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#33FF57")
        chord = Line(points[0].get_center(), points[1].get_center(), color=BLUE)
        self.play(Create(chord))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#3357FF")
        count_label1 = Text("Regions: 1", font_size=24, color=WHITE)
        self.place_at_grid(count_label1, 'E5', scale_factor=0.7)
        self.play(Transform(count_label, count_label1))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FF33A8")
        # Keep 1 point? The storyboard says "One point gives one region"
        count_label1b = Text("Regions: 1", font_size=24, color=WHITE)
        self.place_at_grid(count_label1b, 'E5', scale_factor=0.7)
        self.play(Transform(count_label, count_label1b))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFFF00")
        # 2 points, 2 regions
        count_label2 = Text("Regions: 2", font_size=24, color=WHITE)
        self.place_at_grid(count_label2, 'F5', scale_factor=0.7)
        self.play(Transform(count_label, count_label2))
        self.wait(2)
