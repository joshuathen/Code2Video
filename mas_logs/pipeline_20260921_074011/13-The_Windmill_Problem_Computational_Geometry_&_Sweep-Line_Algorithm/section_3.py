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
            "Hitting a point changes the line's pivot.",
            "This is mathematically a swap of anchor points.",
            "We incrementally update the sorted angular list.",
            "Efficient updates keep collision detection fast.",
            "The mechanism continuously repeats for every hit."
        ]
        self.setup_layout("The Swap Operation", lecture_lines)
        
        # Elements
        pivot_dot = Dot(color=YELLOW)
        point_a = Dot(color=BLUE)
        point_b = Dot(color=RED)
        
        # Initial visual
        self.place_at_grid(pivot_dot, "C4", scale_factor=0.8)
        self.place_at_grid(point_a, "B4", scale_factor=0.8)
        self.place_at_grid(point_b, "D4", scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(pivot_dot), FadeIn(point_a), FadeIn(point_b))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        pivot_line = DashedLine(point_a.get_center(), point_b.get_center(), color=WHITE)
        self.play(Create(pivot_line))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(PURPLE))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(ORANGE))
        self.play(pivot_dot.animate.move_to(point_b.get_center()))
        self.play(FadeOut(pivot_line))
