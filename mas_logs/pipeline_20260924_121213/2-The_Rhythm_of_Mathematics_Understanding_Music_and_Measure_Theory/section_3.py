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
            "Notes represent fractions of a musical measure.",
            "Whole notes equal one full measure capacity.",
            "Quarter notes represent one-fourth of the measure.",
            "Combining notes must sum to the measure total.",
            "Mathematics ensures notes perfectly fill each measure."
        ]
        self.setup_layout("Mathematical Division of Time", lecture_lines)
        
        # Assets (Loaded once)
        metronome = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg")
        clock = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/clock.svg")
        
        # Note shapes
        whole_note = Circle(radius=0.5, color=WHITE).set_fill(WHITE, opacity=0.3)
        h1 = Circle(radius=0.35, color="#FFFF00").set_fill("#FFFF00", opacity=0.3)
        h2 = Circle(radius=0.35, color="#FFFF00").set_fill("#FFFF00", opacity=0.3)
        q1 = Circle(radius=0.25, color="#00FFFF").set_fill("#00FFFF", opacity=0.3)
        q2 = Circle(radius=0.25, color="#00FFFF").set_fill("#00FFFF", opacity=0.3)
        q3 = Circle(radius=0.25, color="#00FFFF").set_fill("#00FFFF", opacity=0.3)
        q4 = Circle(radius=0.25, color="#00FFFF").set_fill("#00FFFF", opacity=0.3)
        e1 = Circle(radius=0.15, color="#FF00FF").set_fill("#FF00FF", opacity=0.3)
        e2 = Circle(radius=0.15, color="#FF00FF").set_fill("#FF00FF", opacity=0.3)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.place_at_grid(metronome, "B6", scale_factor=0.5)
        self.add(metronome)
        self.place_at_grid(whole_note, "B2", scale_factor=0.8)
        self.add(whole_note)
        self.wait(2)
        
        self.play(FadeOut(whole_note))
        self.place_at_grid(h1, "C2", scale_factor=0.8)
        self.place_at_grid(h2, "C4", scale_factor=0.8)
        self.add(h1, h2)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        self.play(FadeOut(h1), FadeOut(h2))
        self.place_at_grid(q1, "D1", scale_factor=0.6)
        self.place_at_grid(q2, "D2", scale_factor=0.6)
        self.place_at_grid(q3, "D3", scale_factor=0.6)
        self.place_at_grid(q4, "D4", scale_factor=0.6)
        self.add(q1, q2, q3, q4)
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF00FF")
        self.play(FadeOut(q1), FadeOut(q2), FadeOut(q3), FadeOut(q4))
        self.place_at_grid(clock, "B5", scale_factor=0.5)
        self.add(clock)
        self.place_at_grid(e1, "C1", scale_factor=0.5)
        self.place_at_grid(e2, "C2", scale_factor=0.5)
        self.add(e1, e2)
        
        # Label Group
        labels = VGroup(Text("Note Types", font_size=24))
        self.place_in_area(labels, "E1", "F6", scale_factor=0.5)
        self.add(labels)
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        self.wait(3)
