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
        # Initial Setup
        title_text = "Prerequisite: The Ladder of Dimensions"
        lecture_lines = [
            "Dimensions build from points to lengths and areas.",
            "A one-dimensional line has length but zero area.",
            "Usually, a wiggling string cannot touch every point."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight line 1 in green
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # 0D: Point
        dot = Dot(color="#00FF00")
        self.place_at_grid(dot, "B3")
        label_0d = Text("0D", font_size=20, color="#00FF00")
        self.place_at_grid(label_0d, "B4")
        
        self.play(FadeIn(dot), Write(label_0d))
        self.wait(1)
        
        # 1D: Length - Transform dot into a line
        line_1d = Line(self.grid["B2"], self.grid["B4"], color="#00FF00")
        label_1d = Text("1D: Length", font_size=20, color="#00FF00")
        self.place_at_grid(label_1d, "B5")
        
        self.play(
            ReplacementTransform(dot, line_1d),
            ReplacementTransform(label_0d, label_1d)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition highlight to line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FF00")
        )
        
        # 2D: Area - Transform line into a square outline
        square_2d = Square(color="#00FF00")
        self.place_in_area(square_2d, "C2", "E4", scale_factor=1.0)
        
        label_2d = Text("2D: Area", font_size=20, color="#00FF00")
        self.place_at_grid(label_2d, "C5")
        
        self.play(
            ReplacementTransform(line_1d, square_2d),
            ReplacementTransform(label_1d, label_2d)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight line 3 in white
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(WHITE)
        )
        
        # Wiggling string (white) inside the square to show it doesn't fill it
        path_points = [
            self.grid["C2"], 
            self.grid["C4"], 
            self.grid["D3"], 
            self.grid["E2"], 
            self.grid["E4"]
        ]
        string_path = VMobject(color=WHITE)
        string_path.set_points_as_corners(path_points).make_smooth()
        
        self.play(Create(string_path))
        self.wait(1)
        
        # Zoom view to emphasize the paradox of zero width/area
        zoom_frame = Square(color=WHITE, stroke_width=2)
        self.place_in_area(zoom_frame, "D5", "F6", scale_factor=0.9)
        
        # In zoom: show a filled green area representing the 2D plane and a thin white line
        zoom_area_fill = Square(color="#00FF00", fill_opacity=0.3, stroke_width=0)
        # Using extremely thin stroke for the line to contrast with the filled area
        zoom_line_detail = Line(LEFT, RIGHT, color=WHITE, stroke_width=0.5)
        zoom_content = VGroup(zoom_area_fill, zoom_line_detail)
        self.place_in_area(zoom_content, "D5", "F6", scale_factor=0.8)
        
        zoom_desc = Text("Zoom: Line has 0 area", font_size=16, color=WHITE)
        # Fix for Issue 28, 29, 30: Move label to bottom area to avoid overlap and improve layout
        self.place_in_area(zoom_desc, "F1", "F4", scale_factor=0.8)
        
        self.play(
            Create(zoom_frame),
            FadeIn(zoom_content),
            Write(zoom_desc)
        )
        self.wait(2)
