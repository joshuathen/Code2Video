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
        # 1. Setup layout with Section Title and Lecture Lines
        title = "Tokenization: Breaking the Code"
        lines = [
            "Computers process numbers, not raw human language.",
            "We chop text into smaller chunks called tokens.",
            "Each token gets a unique ID for processing."
        ]
        self.setup_layout(title, lines)
        
        # Color definitions per line to match animation elements
        COLOR_LINE1 = WHITE
        COLOR_LINE2 = "#FF5555"
        COLOR_LINE3 = "#5555FF"

        # === Animation for Lecture Line 1 ===
        # Lecture: Computers process numbers, not raw human language.
        # Animation: Write 'Unbelievable' in the center (#FFFFFF).
        self.play(self.lecture[0].animate.set_color(COLOR_LINE1))
        
        word = Text("Unbelievable", font_size=42, color=COLOR_LINE1)
        # Fix Issue 44/60: Move 'Unbelievable' word to upper area to avoid clustering
        self.place_in_area(word, "B1", "C6", scale_factor=0.9)
        
        self.play(Write(word))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Lecture: We chop text into smaller chunks called tokens.
        # Animation: Create three red vertical lines (#FF5555) that move through the word at specific character positions.
        self.play(self.lecture[1].animate.set_color(COLOR_LINE2))
        
        # Identify split points for 'Un', 'believ', 'able'
        # indices: U(0) n(1) | b(2) e(3) l(4) i(5) e(6) v(7) | a(8) b(9) l(10) e(11)
        # Calculate slice coordinates between 'n'/'b' and 'v'/'a'
        line1_x = (word[1].get_right()[0] + word[2].get_left()[0]) / 2
        line2_x = (word[7].get_right()[0] + word[8].get_left()[0]) / 2
        y_pos = word.get_center()[1]
        
        slice_line1 = Line(UP*0.7, DOWN*0.7, color=COLOR_LINE2, stroke_width=6).move_to([line1_x, y_pos, 0])
        slice_line2 = Line(UP*0.7, DOWN*0.7, color=COLOR_LINE2, stroke_width=6).move_to([line2_x, y_pos, 0])
        
        # Animate slices dropping through the word
        slice_line1.shift(UP * 2)
        slice_line2.shift(UP * 2)
        
        self.play(
            slice_line1.animate.shift(DOWN * 2),
            slice_line2.animate.shift(DOWN * 2),
            run_time=1
        )
        
        # Define the three segments from the original Text object
        seg1 = word[0:2]   # "Un"
        seg2 = word[2:8]   # "believ"
        seg3 = word[8:]    # "able"
        
        # Visually separate segments to show the effect of the 'cut'
        self.play(
            seg1.animate.shift(LEFT * 0.3),
            seg3.animate.shift(RIGHT * 0.3),
            FadeOut(slice_line1),
            FadeOut(slice_line2)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Lecture: Each token gets a unique ID for processing.
        # Animation: Transform the segments into three rectangles with numbers: [464], [3452], [1205] in light blue (#5555FF).
        self.play(self.lecture[2].animate.set_color(COLOR_LINE3))
        
        # Prepare ID boxes
        id_values = ["464", "3452", "1205"]
        token_boxes = VGroup()
        for val in id_values:
            rect = RoundedRectangle(corner_radius=0.1, height=0.8, width=1.5, color=COLOR_LINE3, fill_opacity=0.1)
            id_txt = Text(val, font_size=28, color=COLOR_LINE3)
            token_boxes.add(VGroup(rect, id_txt))
        
        # Position the group of boxes in the same general area
        token_boxes.arrange(RIGHT, buff=0.5)
        # Fix Issue 45/60: Move token_boxes to lower area to avoid overlap with original word position
        self.place_in_area(token_boxes, "E1", "F6", scale_factor=0.9)
        
        # Transform the word segments into the blue ID token boxes
        self.play(
            ReplacementTransform(seg1, token_boxes[0]),
            ReplacementTransform(seg2, token_boxes[1]),
            ReplacementTransform(seg3, token_boxes[2])
        )
        
        self.wait(3)
