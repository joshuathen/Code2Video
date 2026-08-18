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
        self.setup_layout("The Dimensional Paradox (Prerequisites)", [
            "A 1D line has length, no area.", 
            "A 2D square has area, infinite points.", 
            "Can a 1D line visit every 2D point?"
        ])
        
        # Animations
        # 1. Display text 'Dimensional Paradox' in center with color #FFFFFF.
        paradox_text = Text("Dimensional Paradox", color=WHITE)
        self.place_in_area(paradox_text, 'C2', 'E5', scale_factor=0.6)
        self.play(FadeIn(paradox_text))
        
        # 2. Fade in small circles representing points, color #FF5733, using icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg].
        # Using SVG for point representation as requested by Asset integration issue
        points = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", color="#FF5733")
        self.place_at_grid(points, 'B4', scale_factor=0.8)
        self.play(FadeIn(points))

        # 3. Highlight gaps between points with glowing color #33FF57.
        # Placeholder indicator for gaps
        gaps = Circle(radius=0.1, color="#33FF57", fill_opacity=0.5)
        self.place_at_grid(gaps, "C4")
        self.play(Create(gaps))

        # 4. Transform circles into a line, color #3357FF.
        line = Line(start=self.grid["C2"], end=self.grid["C5"], color="#3357FF", stroke_width=4)
        self.play(Transform(points, line), FadeOut(gaps))
        
        # 5. Show text 'Continuum' appearing above, color #FFFFFF.
        continuum_text = Text("Continuum", color=WHITE, font_size=24)
        self.place_at_grid(continuum_text, 'A4', scale_factor=0.7)
        self.play(Write(continuum_text))

        # Coloring lecture lines
        self.play(self.lecture[0].animate.set_color("#FF5733"))
        self.play(self.lecture[1].animate.set_color("#33FF57"))
        self.play(self.lecture[2].animate.set_color("#3357FF"))
        
        self.wait(2)
