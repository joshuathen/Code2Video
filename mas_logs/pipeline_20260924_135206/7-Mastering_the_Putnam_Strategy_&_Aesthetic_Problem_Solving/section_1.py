from manim import *
import os

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
        lines = [
            "Putnam problems test structural insight, not speed.",
            "Adopt the Discovery Cycle: Observe, Conjecture, Prove.",
            "Visualize logical paths like a detective's map.",
            "Identify symmetry, not just brute-force calculation.",
            "Focus on geometric patterns."
        ]
        self.setup_layout("The Putnam Mindset: Beyond Computational Drills", lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        # Using a default shape if asset missing, but placeholder structure is ready
        placeholder_bulb = Circle(radius=0.3, color=YELLOW)
        self.place_at_grid(placeholder_bulb, 'A5', scale_factor=0.5)
        self.play(FadeIn(placeholder_bulb))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00CED1"))
        cycle = VGroup(Text("Observe", font_size=18), Text("Conjecture", font_size=18), Text("Prove", font_size=18)).arrange(DOWN)
        self.place_at_grid(cycle, 'B5', scale_factor=0.5)
        self.play(Write(cycle))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF6347"))
        detective_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/detective.svg"
        detective = SVGMobject(detective_path) if os.path.exists(detective_path) else Dot(color=BLUE)
        self.place_at_grid(detective, 'C5', scale_factor=0.5)
        self.play(FadeIn(detective))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#90EE90"))
        sym = VGroup(Line(LEFT, RIGHT), Line(UP, DOWN)).rotate(PI/4).set_color(ORANGE)
        self.place_at_grid(sym, 'D5', scale_factor=0.5)
        self.play(Create(sym))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#DDA0DD"))
        squirrel_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/squirrel.svg"
        squirrel = SVGMobject(squirrel_path) if os.path.exists(squirrel_path) else Square(color=ORANGE)
        self.place_at_grid(squirrel, 'E5', scale_factor=0.5)
        self.play(FadeIn(squirrel))
        self.wait(2)
