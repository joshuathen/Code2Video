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
        # Setup the layout with title and lecture lines
        title = "Prerequisite: The Cartesian Stage"
        lines = [
            "Our world begins on a 2D coordinate grid.",
            "A point like the treasure chest is a fixed location.",
            "But vectors describe movement, not just a static spot.",
            "An arrow shows the path from origin to coordinates.",
            "This instruction tells us exactly where to go."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Create a white #FFFFFF coordinate grid and label the x and y axes.
        self.lecture[0].set_color("#FFFFFF")
        
        # Define the plane.
        # Adjusted range and scale to ensure visuals stay within the grid area (B2-F6).
        # Origin (0,0) will be at D4.
        plane = NumberPlane(
            x_range=[-1, 4, 1],
            y_range=[-1, 3, 1],
            background_line_style={
                "stroke_color": "#FFFFFF",
                "stroke_width": 1,
                "stroke_opacity": 0.3
            },
            axis_config={
                "stroke_color": "#FFFFFF",
                "include_tip": True,
            }
        )
        self.place_in_area(plane, "B2", "F6", scale_factor=0.5)
        
        # Axis labels placed at specific grid positions to avoid manual positioning.
        x_axis_label = Text("x", color="#FFFFFF")
        y_axis_label = Text("y", color="#FFFFFF")
        self.place_at_grid(x_axis_label, "D6", scale_factor=0.5) # Right of x-axis
        self.place_at_grid(y_axis_label, "B4", scale_factor=0.5) # Above y-axis
        
        self.play(Create(plane), Write(x_axis_label), Write(y_axis_label))
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # Place a yellow #FFFF00 dot at (3, 2) and label it 'Point (3,2)'.
        self.lecture[1].set_color("#FFFF00")
        
        dot = Dot(plane.c2p(3, 2), color="#FFFF00")
        point_label = Text("Point (3,2)", color="#FFFF00")
        # Fix for Issue 27 & 29: Use specific grid pos and scale_factor=0.5
        self.place_at_grid(point_label, "C5", scale_factor=0.5)
        
        self.play(FadeIn(dot), Write(point_label))
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # Fade out the yellow dot and its label to clear the stage.
        self.lecture[2].set_color("#FFFFFF")
        
        self.play(FadeOut(dot), FadeOut(point_label))
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        # Draw a cyan #00FFFF arrow from (0,0) to (3,2) representing 'Vector [3,2]'.
        self.lecture[3].set_color("#00FFFF")
        
        vector_arrow = Arrow(
            start=plane.c2p(0, 0),
            end=plane.c2p(3, 2),
            buff=0,
            color="#00FFFF",
            stroke_width=4
        )
        # Fix for Issue 28 & 29: Place formula in a safe area with scale_factor=0.6
        vector_formula = MathTex(r"\text{Vector } \begin{bmatrix} 3 \\ 2 \end{bmatrix}", color="#00FFFF")
        self.place_in_area(vector_formula, "C1", "E1", scale_factor=0.6)
        
        self.play(GrowArrow(vector_arrow), Write(vector_formula))
        self.wait(2.0)

        # === Animation for Lecture Line 5 ===
        # Pulse the cyan #00FFFF arrow to emphasize it represents an instruction for movement.
        self.lecture[4].set_color("#00FFFF")
        
        # Using Indicate for pulsing (Lesson L004)
        self.play(Indicate(vector_arrow, color="#00FFFF", scale_factor=1.2))
        self.wait(2.0)
