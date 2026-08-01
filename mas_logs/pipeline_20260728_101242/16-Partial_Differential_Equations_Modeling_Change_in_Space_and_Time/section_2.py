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

class Section2Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title = "Prerequisite Kit: The Partial Derivative"
        lines = [
            "- Partial derivatives measure change while holding other variables constant.",
            "- Imagine hiking a mountain along a single compass direction.",
            "- We isolate one dimension to understand complex multidimensional shapes."
        ]
        
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        # Load mountain SVG [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg]
        try:
            mountain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg")
        except Exception:
            mountain = Triangle().scale(2)
            
        mountain.set_color("#666666")
        # Issue 27 Fix: Adjusted area and scale to avoid overlap with text
        self.place_in_area(mountain, 'A3', 'F6', scale_factor=1.4)
        
        # Small yellow sphere (#FFFF00) representing the hiker
        sphere = Dot(color="#FFFF00", radius=0.1)
        # Issue 28 Fix: Adjusted grid position and scale to reduce visual clutter
        self.place_at_grid(sphere, 'B4', scale_factor=0.8)
        
        # Highlight first lecture line in Yellow (matching sphere)
        self.lecture[0].set_color(YELLOW)
        self.play(
            DrawBorderThenFill(mountain, run_time=2),
            FadeIn(sphere, shift=UP)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Green tangent line (#00FF00) along the x-direction (horizontal-ish)
        tangent_x = Line(
            start=sphere.get_center() + LEFT * 1.0 + DOWN * 0.1,
            end=sphere.get_center() + RIGHT * 1.0 + UP * 0.1,
            color="#00FF00",
            stroke_width=6
        )
        
        # Highlight second lecture line in Green (matching tangent line)
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN)
        self.play(Create(tangent_x))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Cyan tangent line (#00FFFF) along the y-direction (slanted-ish)
        tangent_y = Line(
            start=sphere.get_center() + LEFT * 0.4 + UP * 0.8,
            end=sphere.get_center() + RIGHT * 0.4 + DOWN * 0.8,
            color="#00FFFF",
            stroke_width=6
        )
        
        # Highlight third lecture line in Cyan (matching tangent line)
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FFFF")
        self.play(Create(tangent_y))
        self.wait(2)
