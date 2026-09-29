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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "We transmit data with one bit change.",
            "Mathematics turns impossible odds into certainty.",
            "Optimal structure is the key insight."
        ]
        self.setup_layout("Conclusion: The Power of Optimization", lecture_lines)
        
        # --- Prepare Visuals ---
        # 1. Data icon (using asset as requested)
        node_a = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/node.svg")
        node_b = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/node.svg")
        bit_flow = Arrow(start=node_a.get_right(), end=node_b.get_left(), color=YELLOW)
        data_group = VGroup(node_a, node_b, bit_flow)
        self.place_in_area(data_group, "B4", "C6", scale_factor=0.6)
        
        # 2. Certainty text
        certainty = Text("CERTAINTY", font_size=48, color="#FFFF00")
        self.place_at_grid(certainty, "D2", scale_factor=0.8)
        certainty.set_opacity(0)
        
        # 3. Optimal structure
        structure = VGroup(*[Square(side_length=0.3, color=PURPLE).set_fill(PURPLE, opacity=0.3) for _ in range(9)])
        structure.arrange_in_grid(3, 3, buff=0.1)
        self.place_at_grid(structure, "E5", scale_factor=0.8)
        structure.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(FadeIn(data_group))
        self.play(data_group.animate.shift(RIGHT * 0.5), run_time=1)
        self.play(data_group.animate.shift(LEFT * 0.5), run_time=1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.play(FadeIn(certainty))
        self.play(Indicate(certainty))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(PURPLE))
        self.play(FadeIn(structure))
        self.play(structure.animate.rotate(PI/4))
        self.wait(2)
