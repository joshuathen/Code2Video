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
        self.setup_layout(
            "Prerequisite: The Slope and the Area", 
            [
                "First, remember slope: how steep a line is.",
                "Second, remember area: the space inside a shape.",
                "These two pillars are the foundation of calculus."
            ]
        )
        
        # Colors
        slope_color = "#FFD700"
        area_color = "#00FF7F"
        calc_color = "#FFFFFF"  # Following storyboard's final white color for lines, but using highlight colors during explanation

        # Assets
        pillar_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/pillar.svg"

        # === Animation for Lecture Line 1 ===
        # Display a right triangle representing a roof and highlight/label slope.
        
        # Create triangle
        triangle = Polygon(ORIGIN, [1.5, 0, 0], [0, 1.2, 0], color=slope_color, stroke_width=4)
        
        # Identify hypotenuse points for highlighting
        hyp_start = triangle.get_vertices()[2]
        hyp_end = triangle.get_vertices()[1]
        hypotenuse = Line(hyp_start, hyp_end, color=WHITE, stroke_width=6)
        
        # Labels
        slope_label = Text("Slope: Rise / Run", font_size=20, color=slope_color)
        
        # Group and position (Resolving Issue 28)
        triangle_group = VGroup(triangle, hypotenuse)
        self.place_in_area(triangle_group, "B2", "D3")
        self.place_in_area(slope_label, "A2", "A3", scale_factor=0.8)

        # Start sequence
        self.play(
            self.lecture[0].animate.set_color(slope_color),
            Create(triangle),
            run_time=1.5
        )
        self.play(
            Create(hypotenuse),
            Write(slope_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display a rectangle and fill it to show area.
        
        # Create rectangle
        rectangle = Rectangle(width=1.5, height=1.2, color=area_color, fill_opacity=0.3)
        # Position (Resolving Issue 29)
        self.place_in_area(rectangle, "B5", "D6")
        
        # Label (Resolving Issue 30)
        area_label = Text("Area: Base x Height", font_size=20, color=area_color)
        self.place_in_area(area_label, "A5", "A6", scale_factor=0.8)
        
        self.play(
            self.lecture[1].animate.set_color(area_color),
            Create(rectangle),
            Write(area_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Emphasize both as pillars of calculus using the pillar asset (Resolving Issue 23).
        
        # Load assets
        pillar1 = SVGMobject(pillar_path, color=GRAY, fill_opacity=0.5)
        pillar2 = SVGMobject(pillar_path, color=GRAY, fill_opacity=0.5)
        
        # Position pillars under the shapes
        self.place_at_grid(pillar1, "E2", scale_factor=0.5)
        self.place_at_grid(pillar2, "E5", scale_factor=0.5)
        
        # Move shapes slightly if needed? No, let's just show pillars.
        # Storyboard says "Show both shapes side-by-side as 'pillars'".
        
        self.play(
            self.lecture[2].animate.set_color(calc_color),
            FadeIn(pillar1),
            FadeIn(pillar2),
            triangle_group.animate.shift(UP * 0.2),
            rectangle.animate.shift(UP * 0.2),
            run_time=1.5
        )
        
        # Highlight all
        self.play(
            triangle.animate.set_stroke(width=6),
            rectangle.animate.set_stroke(width=6),
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(WHITE),
            run_time=1
        )
        
        self.wait(2)
