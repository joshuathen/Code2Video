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
            "Existence proofs often rely on extremes.",
            "Pigeonhole Principle forces structural overlap.",
            "Extremal thinking simplifies complex constraints."
        ]
        self.setup_layout("Tactical Tool: The Pigeonhole Principle & Extremal Thinking", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Show title: Pigeonhole & Extremal (#FFD700) fade in.
        title_highlight = Text("Pigeonhole & Extremal", font_size=32, color="#FFD700")
        self.place_at_grid(title_highlight, 'A4', scale_factor=0.8)
        self.play(FadeIn(title_highlight))
        self.play(self.lecture[0].animate.set_color("#FFD700"))

        # === Animation for Lecture Line 2 ===
        # Animate [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pigeon.svg] objects filling pigeonholes (#00CED1).
        holes = VGroup(*[Square(side_length=0.6, color="#00CED1") for _ in range(4)])
        holes.arrange(RIGHT, buff=0.2)
        self.place_at_grid(holes, 'C3', scale_factor=0.9)
        
        # Using SVG asset for pigeons
        pigeon_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pigeon.svg", color="#FFFFFF")
        pigeons = VGroup(*[pigeon_img.copy().scale(0.3) for _ in range(5)])
        
        self.play(FadeIn(holes))
        for p in pigeons:
            self.play(FadeIn(p.move_to(holes[0].get_center())), run_time=0.3)
        self.play(self.lecture[1].animate.set_color("#00CED1"))

        # === Animation for Lecture Line 3 ===
        # Highlight maximum/minimum elements in #FF4500.
        extremes = VGroup(*[Dot(color="#FF4500") for _ in range(5)])
        self.place_at_grid(extremes, 'D4', scale_factor=0.9)
        
        # Highlight the largest dot
        max_dot = extremes[4].copy().set_color("#FF4500").scale(1.5)
        self.play(FadeIn(extremes), FadeIn(max_dot))
        self.play(self.lecture[2].animate.set_color("#FF4500"))
        
        self.wait(2)
