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
        # Data from shared state
        title_text = "Prerequisite: The Coordinate Plane (The Vector's Home)"
        lecture_lines = [
            "In a grid, vectors usually start at the origin.",
            "A vector points to a specific coordinate pair.",
            "This vector moves three right and two up."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Hex colors as per storyboard/instruction
        COLOR_GRID = "#555555"
        COLOR_VECTOR = "#FFFF00"
        
        # === Animation for Lecture Line 1 ===
        # Fade in a gray coordinate grid (#555555).
        plane = NumberPlane(
            x_range=[-2, 5, 1],
            y_range=[-2, 4, 1],
            background_line_style={
                "stroke_color": COLOR_GRID,
                "stroke_width": 2,
                "stroke_opacity": 0.6
            },
            axis_config={"color": COLOR_GRID}
        )
        # Position plane origin at D2 to provide more space as requested by Issue 24.
        self.place_at_grid(plane, 'D2', scale_factor=0.7)
        
        self.play(
            self.lecture[0].animate.set_color(COLOR_GRID),
            FadeIn(plane)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A yellow arrow (#FFFF00) grows from (0,0) to (3,2).
        vec_start = plane.c2p(0, 0)
        vec_end = plane.c2p(3, 2)
        vector = Arrow(vec_start, vec_end, buff=0, color=COLOR_VECTOR)
        
        self.play(
            self.lecture[1].animate.set_color(COLOR_VECTOR),
            GrowArrow(vector)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The coordinates '[3, 2]' (#FFFF00) appear at the arrow's tip.
        # Positioned at B6 as requested by Issue 25 to avoid overlap.
        coords_label = MathTex(r"\begin{bmatrix} 3 \\ 2 \end{bmatrix}", color=COLOR_VECTOR)
        self.place_at_grid(coords_label, 'B6', scale_factor=0.9)
        
        self.play(
            self.lecture[2].animate.set_color(COLOR_VECTOR),
            Write(coords_label)
        )
        self.wait(2)
