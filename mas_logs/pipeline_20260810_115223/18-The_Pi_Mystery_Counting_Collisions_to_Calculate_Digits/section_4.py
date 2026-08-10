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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Mass ratios define the path angle.", 
            "Paths measure the circle's arc.", 
            "Pi emerges from these reflections."
        ]
        self.setup_layout("The Emergence of Pi", lecture_lines)
        
        # Color definitions for lecture lines
        colors = ["#FFFFFF", "#FFD700", "#00FF00"]

        # === Animation for Lecture Line 1 ===
        # Mass ratios define the path angle
        circle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg", color=colors[0])
        # Note: Storyboard requires circle formula for lecture 1, icon used as placeholder
        self.place_at_grid(circle_icon, 'B4', scale_factor=0.9)
        self.play(FadeIn(circle_icon))
        self.lecture[0].set_color(colors[0])
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Paths measure the circle's arc
        circle_icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg", color=colors[1])
        radius_line = Line(ORIGIN, RIGHT * 1.5, color=colors[1])
        radius_group = VGroup(circle_icon2, radius_line)
        self.place_in_area(radius_group, 'B5', 'C6', scale_factor=0.7)
        self.play(Create(radius_group))
        self.lecture[1].set_color(colors[1])
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Pi emerges from these reflections
        convergence_highlight = Circle(radius=1.0, color=colors[2], fill_opacity=0.3)
        self.place_at_grid(convergence_highlight, 'E5', scale_factor=0.8)
        self.play(FadeIn(convergence_highlight))
        self.lecture[2].set_color(colors[2])
        self.wait(2)
