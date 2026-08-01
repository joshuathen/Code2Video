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
        # Setup the scene layout with title and lecture lines
        title_text = "Prerequisites: Vectors as Coordinates"
        lecture_lines = [
            "A vector is an arrow starting from the origin.",
            "We represent it using coordinates, like x and y.",
            "Basis vectors i-hat and j-hat are our building blocks."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight the first lecture line
        self.lecture[0].set_color(WHITE)
        
        # Create a coordinate grid. 
        # NumberPlane uses Text for labels by default in modern Manim, 
        # but we use it here without labels to avoid any LaTeX dependency.
        plane = NumberPlane(
            x_range=[-2, 5, 1],
            y_range=[-2, 4, 1],
            background_line_style={"stroke_color": "#444444"},
            axis_config={"stroke_color": "#444444"}
        )
        self.place_at_grid(plane, "D3")
        
        # Create a white arrow from (0,0) to (3,2).
        arrow = Arrow(
            start=plane.c2p(0, 0),
            end=plane.c2p(3, 2),
            buff=0,
            color="#FFFFFF",
            stroke_width=4
        )
        
        self.play(FadeIn(plane))
        self.play(GrowArrow(arrow))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight the second lecture line.
        self.lecture[1].set_color(WHITE)
        
        # Display coordinates [3, 2] using Text instead of MathTex to avoid LaTeX requirement
        coords = Text("[3, 2]", color="#FFFFFF", font_size=24)
        # Fix for Issue 27: Adjusted scale factor to 0.75 to avoid cramping
        self.place_at_grid(coords, "A6", scale_factor=0.75)

        self.play(Write(coords))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight the third lecture line
        self.lecture[2].set_color(YELLOW)
        
        # Create unit vectors i-hat (#FF0000) and j-hat (#00FF00).
        i_hat = Arrow(
            start=plane.c2p(0, 0),
            end=plane.c2p(1, 0),
            buff=0,
            color="#FF0000",
            stroke_width=6
        )
        j_hat = Arrow(
            start=plane.c2p(0, 0),
            end=plane.c2p(0, 1),
            buff=0,
            color="#00FF00",
            stroke_width=6
        )
        
        # Labels for unit vectors using Text with slant=ITALIC to represent math variables.
        i_label = Text("i", slant=ITALIC, color="#FF0000", font_size=24)
        j_label = Text("j", slant=ITALIC, color="#00FF00", font_size=24)
        
        # Position labels at adjacent grid cells for clarity.
        # Fix for Issue 26: Confirmed i_label at E4
        self.place_at_grid(i_label, "E4", scale_factor=0.8) 
        # Fix for Issue 25: Moved j_label to B3 to be directly above the unit vector tip
        self.place_at_grid(j_label, "B3", scale_factor=0.8) 
        
        self.play(
            GrowArrow(i_hat),
            GrowArrow(j_hat),
            FadeIn(i_label),
            FadeIn(j_label)
        )
        self.wait(2)
