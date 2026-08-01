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
        # Setup the layout with title and lecture lines
        self.setup_layout(
            "Prerequisite: The Language of Vectors", 
            [
                "Computers process numbers, not raw text.",
                "Embeddings map words into a multi-dimensional space.",
                "Related concepts sit closer together in this space."
            ]
        )
        
        # Define colors
        KING_COLOR = "#4169E1"
        QUEEN_COLOR = "#FF69B4"
        CAR_COLOR = "#808080"
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        # computers process numbers, not raw text.
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        # Create a grid background for the animation area
        grid_lines = VGroup()
        for i in range(7):
            # Vertical lines
            start_v = self.grid["A1"] + LEFT * 0.5 + UP * 0.5 + RIGHT * i
            end_v = self.grid["F1"] + LEFT * 0.5 + DOWN * 0.5 + RIGHT * i
            grid_lines.add(Line(start_v, end_v, stroke_width=1, color=GRAY_D))
            # Horizontal lines
            start_h = self.grid["A1"] + LEFT * 0.5 + UP * 0.5 + DOWN * i
            end_h = self.grid["A6"] + RIGHT * 0.5 + UP * 0.5 + DOWN * i
            grid_lines.add(Line(start_h, end_h, stroke_width=1, color=GRAY_D))
        
        king_dot = Dot(color=KING_COLOR)
        king_label = Text("King", font_size=18, color=KING_COLOR).next_to(king_dot, UP, buff=0.1)
        king_group = VGroup(king_dot, king_label)
        # Fix Issue 28: Scale factor 0.7
        self.place_at_grid(king_group, "B2", scale_factor=0.7)

        queen_dot = Dot(color=QUEEN_COLOR)
        queen_label = Text("Queen", font_size=18, color=QUEEN_COLOR).next_to(queen_dot, UP, buff=0.1)
        queen_group = VGroup(queen_dot, queen_label)
        # Fix Issue 26: Move to B4 and Scale factor 0.7
        self.place_at_grid(queen_group, "B4", scale_factor=0.7)

        self.play(Create(grid_lines), FadeIn(king_group), FadeIn(queen_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Embeddings map words into a multi-dimensional space.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        car_dot = Dot(color=CAR_COLOR)
        car_label = Text("Car", font_size=18, color=CAR_COLOR).next_to(car_dot, UP, buff=0.1)
        car_group = VGroup(car_dot, car_label)
        # Fix Issue 27: Move to F6 and Scale factor 0.7
        self.place_at_grid(car_group, "F6", scale_factor=0.7)

        # Arrow highlighting distance between King and Car
        dist_arrow = DoubleArrow(
            king_dot.get_center(), 
            car_dot.get_center(), 
            buff=0.1, 
            color=WHITE, 
            stroke_width=2,
            tip_length=0.15
        )
        dist_label = Text("Large Distance", font_size=14, color=WHITE).move_to(dist_arrow.get_center() + UP * 0.3)

        self.play(FadeIn(car_group))
        self.play(GrowArrow(dist_arrow), Write(dist_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Related concepts sit closer together in this space.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR),
            FadeOut(dist_arrow),
            FadeOut(dist_label)
        )

        # Flash King and Queen
        self.play(Indicate(king_dot), Indicate(queen_dot))

        # Relationship vector (line)
        rel_line = Line(king_dot.get_center(), queen_dot.get_center(), color=WHITE, stroke_width=2)
        rel_label = Text("Semantic Similarity", font_size=14, color=WHITE).next_to(rel_line, DOWN, buff=0.1)

        self.play(Create(rel_line), Write(rel_label))
        self.wait(3)

        # Cleanup highlight
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
