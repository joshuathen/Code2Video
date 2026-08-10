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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Map binary counters directly to moves.",
            "Changing bits tell us which disk moves.",
            "Rightmost bit changes mean smallest disk moves.",
            "Middle bit changes mean second disk moves.",
            "Binary state dictates every physical shift."
        ]
        self.setup_layout("The Core Connection: Binary Bits as Disk Moves", lecture_lines)
        
        # Color palette for lecture lines
        colors = [BLUE, GREEN, YELLOW, ORANGE, RED]
        
        # Setup visual elements
        binary_display = VGroup(*[Text(f"{i}: {bin(i)[2:].zfill(3)}", font_size=24) for i in range(1, 8)])
        binary_display.arrange(DOWN, buff=0.2)
        
        state_summary_group = VGroup(Text("Binary State Map", font_size=20), binary_display).arrange(DOWN, buff=0.3)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        self.place_in_area(binary_display, 'B3', 'E4', scale_factor=0.75)
        self.play(FadeIn(binary_display))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(colors[3]))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(colors[4]))
        self.place_in_area(state_summary_group, 'B5', 'F6', scale_factor=0.8)
        self.play(FadeOut(binary_display), FadeIn(state_summary_group))
