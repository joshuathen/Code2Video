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

        # Manim provides FRAME_WIDTH and FRAME_HEIGHT constants
        # We should import them or use them as provided by the framework
        frame_width = config.frame_width
        frame_height = config.frame_height

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Adjusted x and y coordinates to center the grid more appropriately
                x = frame_width / 4 + j * frame_width / 12
                y = frame_height / 4 - i * frame_height / 12
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

class Section7Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "DP-3T effectively traces contacts during outbreaks.",
            "It strongly upholds user privacy principles.",
            "This makes it a vital digital health tool.",
            "Balancing public health and privacy is key.",
            "DP-3T offers a robust solution."
        ]
        self.setup_layout("Key Takeaways and Benefits", lecture_lines)
        
        # Define colors for lecture lines
        colors = [
            "#FF6B6B",  # Red
            "#4ECDC4",  # Turquoise
            "#45B7D1",  # Blue
            "#FED766",  # Yellow
            "#A0A0A0"   # Grey
        ]

        # === Animation for Lecture Line 1 ===
        # DP-3T effectively traces contacts during outbreaks.
        lecture_line_1 = self.lecture[0]
        lecture_line_1.set_color(colors[0])
        # No animation needed as per instructions, only color change is applied

        # === Animation for Lecture Line 2 ===
        # It strongly upholds user privacy principles.
        lecture_line_2 = self.lecture[1]
        lecture_line_2.set_color(colors[1])
        # No animation needed as per instructions, only color change is applied

        # === Animation for Lecture Line 3 ===
        # This makes it a vital digital health tool.
        lecture_line_3 = self.lecture[2]
        lecture_line_3.set_color(colors[2])
        # No animation needed as per instructions, only color change is applied

        # === Animation for Lecture Line 4 ===
        # Balancing public health and privacy is key.
        lecture_line_4 = self.lecture[3]
        lecture_line_4.set_color(colors[3])
        # No animation needed as per instructions, only color change is applied

        # === Animation for Lecture Line 5 ===
        # DP-3T offers a robust solution.
        lecture_line_5 = self.lecture[4]
        lecture_line_5.set_color(colors[4])
        # No animation needed as per instructions, only color change is applied

        self.wait(1)