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
        # Data from shared state
        title_text = "Vectors as Movements, Not Just Numbers"
        lecture_lines = [
            "Meet Pixie, our guide in this digital grid world.",
            "Vectors are arrows showing movement, not just static numbers.",
            "Vector [3, 2] means moving 3 right and 2 up."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors based on storyboard and teaching flow
        COLOR_PIXIE = "#FF00FF"  # Magenta for our character
        COLOR_VECTOR = "#00FFFF" # Cyan for the vector arrow
        COLOR_LABEL = "#FFFF00"  # Yellow for text/numbers and lecture highlights
        COLOR_DASH = WHITE

        # === Animation for Lecture Line 1 ===
        # Step 1: Create a black background with a white grid. Introduce the vector arrow.
        # Issue 26: Use larger scale and full right-side area
        plane = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 4, 1],
            background_line_style={"stroke_opacity": 0.3},
            axis_config={"include_numbers": True, "stroke_color": WHITE}
        )
        self.place_in_area(plane, "A1", "F6", scale_factor=0.8)
        
        vector = Arrow(
            start=plane.coords_to_point(0, 0),
            end=plane.coords_to_point(3, 2),
            buff=0,
            color=COLOR_VECTOR,
            stroke_width=6
        )

        self.play(self.lecture[0].animate.set_color(COLOR_LABEL))
        self.play(Create(plane))
        self.play(GrowArrow(vector))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Step 2: Display text "[3, 2]" near vector tip and dashed projection lines.
        # Issue 27: Increase label scale to 1.0
        label = Text("[3, 2]", font_size=24, color=COLOR_LABEL)
        self.place_at_grid(label, "B5", scale_factor=1.0)
        
        dash_x = DashedLine(
            plane.coords_to_point(3, 0), 
            plane.coords_to_point(3, 2), 
            color=COLOR_DASH
        )
        dash_y = DashedLine(
            plane.coords_to_point(0, 2), 
            plane.coords_to_point(3, 2), 
            color=COLOR_DASH
        )

        self.play(self.lecture[1].animate.set_color(COLOR_LABEL))
        self.play(Write(label), Create(dash_x), Create(dash_y))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Step 3: Animate Pixie moving along the x-axis then y-axis to (3,2).
        pixie = Circle(radius=0.1, color=COLOR_PIXIE, fill_opacity=1.0)
        pixie.move_to(plane.coords_to_point(0, 0))
        
        self.play(self.lecture[2].animate.set_color(COLOR_LABEL))
        self.play(FadeIn(pixie))
        
        # Movement along x-axis then y-axis
        self.play(
            pixie.animate.move_to(plane.coords_to_point(3, 0)), 
            run_time=1.5, 
            rate_func=linear
        )
        self.play(
            pixie.animate.move_to(plane.coords_to_point(3, 2)), 
            run_time=1.0, 
            rate_func=linear
        )
        
        self.wait(3)
