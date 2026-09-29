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
        self.setup_layout("Conclusion and Exploration", [
            "Mathematics is the study of deep patterns.",
            "Ulam spirals reveal the hidden order.",
            "Everything connects to fundamental constants."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Recap core journey from chaos to structure
        dot_cloud = VGroup(*[Dot(radius=0.03, color=BLUE) for _ in range(50)])
        for dot in dot_cloud:
            dot.move_to(self.grid["B3"] + np.random.uniform(-1, 1, 3))
        
        ulam_spiral = VGroup(*[Dot(radius=0.03, color=YELLOW) for _ in range(50)])
        angle = np.linspace(0, 4*PI, 50)
        for i, dot in enumerate(ulam_spiral):
            r = angle[i] * 0.1
            dot.move_to(self.grid["B3"] + np.array([r*np.cos(angle[i]), r*np.sin(angle[i]), 0]))
        self.place_in_area(ulam_spiral, 'A2', 'C5', scale_factor=0.9)
            
        self.play(FadeIn(dot_cloud))
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Transform(dot_cloud, ulam_spiral))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Show lingering connection lines fading into background
        lines = VGroup()
        for i in range(len(ulam_spiral)-1):
            line = Line(ulam_spiral[i].get_center(), ulam_spiral[i+1].get_center(), color=WHITE, stroke_width=1)
            lines.add(line)
        self.play(Create(lines), run_time=2)
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(lines.animate.set_opacity(0.3))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Present closing question
        closing_text = Text("Is there an ultimate constant?", font_size=30, color=WHITE)
        closing_text.set_stroke(color=ORANGE, width=2)
        self.place_in_area(closing_text, 'D2', 'D5', scale_factor=0.85)
        
        self.play(Write(closing_text))
        self.play(self.lecture[2].animate.set_color(ORANGE))
        self.wait(2)
