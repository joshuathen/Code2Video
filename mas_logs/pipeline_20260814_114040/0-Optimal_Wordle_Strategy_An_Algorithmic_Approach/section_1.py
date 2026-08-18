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
        lecture_lines = ["Wordle is a strategic search problem.", "Each guess is a query for information.", "The goal is reducing the search space."]
        self.setup_layout("The Game Mechanics: Information Theory Basics", lecture_lines)
        
        # Mobjects
        wordle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wordle.svg", color=WHITE)
        circle = Circle(radius=1.5, color="#FF00FF")
        dots = VGroup(*[Dot(color="#00FFFF") for _ in range(20)]).arrange_in_grid(4, 5, buff=0.2)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.lecture[0].set_color("#FFFF00")
        self.place_at_grid(wordle_icon, 'C2', scale_factor=0.5)
        self.play(FadeIn(wordle_icon))
        
        self.place_in_area(circle, 'C2', 'F5', scale_factor=0.6)
        self.play(Create(circle))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.lecture[1].set_color("#FFFF00")
        self.place_in_area(dots, 'D3', 'E4', scale_factor=0.5)
        self.play(FadeIn(dots))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.lecture[2].set_color("#FFFF00")
        # Visual highlighting entropy/gain
        highlight = Circle(radius=0.5, color="#FFFF00").set_fill(opacity=0.3)
        self.place_at_grid(highlight, 'C2', scale_factor=0.7)
        self.play(Create(highlight))
        self.wait(2)
