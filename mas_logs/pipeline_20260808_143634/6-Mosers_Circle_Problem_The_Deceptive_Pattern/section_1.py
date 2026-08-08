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
        self.setup_layout("Moser's Circle Problem", [
            "Connect points on a circle with chords.",
            "To maximize regions, avoid triple intersections.",
            "This is Moser's Circle Problem."
        ])
        
        # --- Animation for Lecture Line 1 ---
        # \"Connect points on a circle with chords.\"
        self.lecture[0].set_color("#FF5733")
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")
        self.place_at_grid(circle, 'C5', scale_factor=0.6)
        self.play(Create(circle))
        
        # Add points and chords
        dot1 = Dot(color=WHITE)
        self.place_at_grid(dot1, 'B5')
        dot2 = Dot(color=WHITE)
        self.place_at_grid(dot2, 'D6')
        dot3 = Dot(color=WHITE)
        self.place_at_grid(dot3, 'D4')
        dot4 = Dot(color=WHITE)
        self.place_at_grid(dot4, 'B3')
        
        chord1 = Line(dot1.get_center(), dot3.get_center(), color=WHITE)
        chord2 = Line(dot2.get_center(), dot4.get_center(), color=WHITE)
        
        self.play(Create(dot1), Create(dot2), Create(dot3), Create(dot4))
        self.play(Create(chord1), Create(chord2))

        # --- Animation for Lecture Line 2 ---
        # \"To maximize regions, avoid triple intersections.\"
        self.lecture[1].set_color("#33FF57")
        self.lecture[0].set_color(WHITE)
        
        # Highlight intersection
        intersection = Dot(color="#FFFF00")
        self.place_at_grid(intersection, 'C5', scale_factor=0.3)
        
        c1_label = Text("C1", font_size=20, color="#FF0000")
        self.place_at_grid(c1_label, 'B4', scale_factor=0.5)
        
        c2_label = Text("C2", font_size=20, color="#00FF00")
        self.place_at_grid(c2_label, 'D4', scale_factor=0.5)
        
        self.play(Write(c1_label), Write(c2_label), Indicate(intersection, scale_factor=2))
        self.play(Create(intersection))

        # --- Animation for Lecture Line 3 ---
        # \"This is Moser's Circle Problem.\"
        self.lecture[2].set_color("#5733FF")
        self.lecture[1].set_color(WHITE)
        
        # Fade out everything except the intersection point
        fade_group = VGroup(circle, dot1, dot2, dot3, dot4, chord1, chord2, c1_label, c2_label)
        self.play(FadeOut(fade_group))
        self.play(Indicate(intersection, scale_factor=3))
        self.wait(1)
