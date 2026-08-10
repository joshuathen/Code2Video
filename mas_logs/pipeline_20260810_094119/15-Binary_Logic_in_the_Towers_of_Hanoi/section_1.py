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
        self.setup_layout("Prerequisite: The Binary Language", [
            "Binary systems use only two digits: zero and one.",
            "Each position represents an increasing power of two.",
            "Four bits can represent numbers from zero to fifteen.",
            "Binary is the language of all modern computers."
        ])
        
        # Elements
        bulb_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/bulb.svg"
        bulb = SVGMobject(bulb_asset)
        bits = Text("001", font_size=48, color=WHITE)
        # Combine into group for better management
        binary_display = VGroup(bits, bulb).arrange(DOWN)
        
        # Applying VideoCritic adjustment: Use C5 for initial position
        self.place_at_grid(binary_display, 'C5', scale_factor=1.2)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE), Write(binary_display))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN), bits.animate.set_color(GOLD))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        bits_new = Text("010", font_size=48, color=WHITE)
        # Applying VideoCritic adjustment: Use B5 for new position
        self.place_at_grid(bits_new, 'B5', scale_factor=1.0)
        self.play(self.lecture[2].animate.set_color(YELLOW), Transform(bits, bits_new))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(RED))
        self.wait(2)
