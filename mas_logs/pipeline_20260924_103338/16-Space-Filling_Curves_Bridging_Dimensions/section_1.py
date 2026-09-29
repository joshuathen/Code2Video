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
        self.setup_layout("The Dimensional Paradox", [
            "A line segment cannot cover an area.",
            "Yet, space-filling curves challenge this boundary.",
            "These curves visit every point in space.",
            "We can map 1D paths to 2D regions.",
            "Continuous lines create surface density."
        ])
        
        # Load thread asset
        thread_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/thread.svg"
        thread_obj = SVGMobject(thread_path)
        
        # === Animation for Lecture Line 1 ===
        line = Line(start=LEFT, end=RIGHT, color=WHITE)
        self.place_at_grid(line, "C2", scale_factor=0.6)
        self.play(Create(line))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        square = Square(side_length=2.0, color="#FF00FF")
        self.place_in_area(square, "C3", "D4", scale_factor=0.8)
        self.play(Transform(line, square))
        self.lecture[1].set_color("#FF00FF")

        # === Animation for Lecture Line 3 ===
        points = VGroup(*[Dot(radius=0.03, color="#00FFFF") for _ in range(50)])
        for p in points:
            p.move_to(square.get_center() + np.random.uniform(-0.6, 0.6, 3))
        self.play(FadeIn(points))
        self.lecture[2].set_color("#00FFFF")

        # === Animation for Lecture Line 4 ===
        label = Text("Map: 1D -> 2D", font_size=20, color=WHITE)
        self.place_at_grid(label, "B3", scale_factor=0.7)
        self.play(Write(label))
        self.lecture[3].set_color(GREEN)

        # === Animation for Lecture Line 5 ===
        # Represent the thread as an evolving curve
        self.place_at_grid(thread_obj, "C4", scale_factor=0.4)
        self.play(FadeIn(thread_obj), run_time=1)
        self.play(thread_obj.animate.set_color("#00FF00"), run_time=1)
        self.lecture[4].set_color("#00FF00")
