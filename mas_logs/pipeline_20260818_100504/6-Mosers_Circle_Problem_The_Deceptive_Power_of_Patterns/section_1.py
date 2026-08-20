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
        self.setup_layout("Introduction & Prerequisite Concept", [
            "Place points on a circle.", 
            "Connect every pair with chords.", 
            "How many regions are created?", 
            "We define this number as R.", 
            "Let us count for small n."
        ])
        
        # Load Assets
        circle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg", color=WHITE)
        points = VGroup(*[Dot(radius=0.05, color=WHITE) for _ in range(5)])
        # Manually arrange points in a circle since arrange() doesn't support 'circle=True'
        for i, dot in enumerate(points):
            angle = i * 2 * PI / len(points)
            dot.move_to(0.5 * (np.cos(angle) * RIGHT + np.sin(angle) * UP))
        
        # Group
        circle_group = VGroup(circle_icon, points)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF5733")
        self.place_in_area(circle_group, 'A3', 'C5', scale_factor=0.9)
        self.play(FadeIn(circle_group))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#33FF57")
        # For simplicity, represent chords as lines between the points
        chords = VGroup()
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                chords.add(Line(points[i].get_center(), points[j].get_center(), color="#FF5733", stroke_width=2))
        self.play(Create(chords))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF33")
        q = Text("?", font_size=32, color="#FFFF33")
        self.place_at_grid(q, "B5", scale_factor=1.0)
        self.play(FadeIn(q))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#5733FF")
        label = Text("R", font_size=24, color="#33FF57")
        self.place_at_grid(label, "A4", scale_factor=0.8)
        self.play(FadeIn(label))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF33A8")
        self.wait(1)
