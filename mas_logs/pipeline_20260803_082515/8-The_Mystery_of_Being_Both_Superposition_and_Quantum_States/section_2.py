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
        # Initialize the layout with the specific title and lecture lines
        self.setup_layout("Prerequisite: The Vector Map", [
            "We can map quantum possibilities using 2D geometry.",
            "Outcomes are represented as axes on a graph.",
            "A quantum state is an arrow within this space."
        ])
        
        # Define common colors
        COLOR_HEADS = "#FFFF00"
        COLOR_TAILS = "#00FFFF"
        COLOR_INACTIVE = GRAY
        COLOR_ACTIVE = WHITE

        # Set initial state for lecture lines
        for i in range(1, len(self.lecture)):
            self.lecture[i].set_color(COLOR_INACTIVE)
        
        # Define key points on the grid to avoid clipping (Issues 37 & 38)
        # Origin and tips adjusted to accommodate labels at B2 and E5
        origin = self.grid["E2"]
        y_tip = self.grid["C2"]
        x_tip = self.grid["E4"]

        # === Animation for Lecture Line 1 ===
        # Line: "We can map quantum possibilities using 2D geometry."
        # Action: Draw the coordinate system axes.
        
        # Create white axes initially
        y_axis = Arrow(start=origin, end=y_tip, buff=0, color=WHITE, stroke_width=4)
        x_axis = Arrow(start=origin, end=x_tip, buff=0, color=WHITE, stroke_width=4)
        
        self.play(Create(y_axis), Create(x_axis))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line: "Outcomes are represented as axes on a graph."
        # Action: Label the axes and update their colors to match the outcomes.
        
        self.play(
            self.lecture[0].animate.set_color(COLOR_INACTIVE),
            self.lecture[1].animate.set_color(COLOR_ACTIVE)
        )
        
        # Create Dirac notation labels
        y_label = MathTex(r"|Heads\rangle", color=COLOR_HEADS)
        x_label = MathTex(r"|Tails\rangle", color=COLOR_TAILS)
        
        # Position labels using grid as requested in Issues 37 & 38
        # y_label at B2 (above y_tip C2)
        # x_label at E5 (to the right of x_tip E4)
        self.place_at_grid(y_label, "B2", scale_factor=0.6)
        self.place_at_grid(x_label, "E5", scale_factor=0.6)
        
        self.play(
            y_axis.animate.set_color(COLOR_HEADS),
            x_axis.animate.set_color(COLOR_TAILS),
            Write(y_label),
            Write(x_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: "A quantum state is an arrow within this space."
        # Action: Draw the state vectors (heads and tails) clearly as white arrows.
        
        self.play(
            self.lecture[1].animate.set_color(COLOR_INACTIVE),
            self.lecture[2].animate.set_color(COLOR_ACTIVE)
        )
        
        # State vector for |Heads> (pointing up)
        v_heads = Arrow(start=origin, end=y_tip, buff=0, color=WHITE, stroke_width=8)
        # State vector for |Tails> (pointing right)
        v_tails = Arrow(start=origin, end=x_tip, buff=0, color=WHITE, stroke_width=8)
        
        # Show the vectors being born from the origin
        self.play(Create(v_heads))
        self.wait(0.5)
        self.play(Create(v_tails))
        
        self.wait(2)
