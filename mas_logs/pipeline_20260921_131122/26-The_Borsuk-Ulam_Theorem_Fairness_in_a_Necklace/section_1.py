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
        lecture_lines = [
            "Imagine a sphere, like a globe.",
            "Antipodal points are exactly opposite.",
            "Like Tokyo and the South Atlantic.",
            "They are mirrors through the center.",
            "Symmetry is built into our world."
        ]
        self.setup_layout("Prerequisites: The Intuition of Antipodes", lecture_lines)
        
        # Load Assets
        globe_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/globe.svg")
        tokyo_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tokyo.svg")
        
        # Prepare Visuals
        self.place_in_area(globe_icon, 'B3', 'E5', scale_factor=1.0)
        
        # Points
        point_tokyo = Dot(color="#FF00FF").move_to(globe_icon.get_center() + RIGHT*0.5 + UP*0.3)
        point_atlantic = Dot(color="#FF00FF").move_to(globe_icon.get_center() + LEFT*0.5 + DOWN*0.3)
        line = Line(point_tokyo.get_center(), point_atlantic.get_center(), color="#00FFFF")
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(globe_icon))
        self.lecture[0].set_color(YELLOW)
        
        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(point_tokyo), FadeIn(point_atlantic))
        self.lecture[1].set_color(YELLOW)
        
        # === Animation for Lecture Line 3 ===
        self.place_at_grid(tokyo_icon, 'E2', scale_factor=0.5)
        self.play(Create(line), FadeIn(tokyo_icon))
        self.lecture[2].set_color(YELLOW)
        
        # === Animation for Lecture Line 4 ===
        self.play(Indicate(point_tokyo, color="#FF0000"), Indicate(point_atlantic, color="#FF0000"))
        self.lecture[3].set_color(YELLOW)
        
        # === Animation for Lecture Line 5 ===
        self.play(Rotate(globe_icon, angle=PI/2), run_time=2)
        self.lecture[4].set_color(YELLOW)
        self.wait(1)
