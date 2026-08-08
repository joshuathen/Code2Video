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
            "The Query searches for information we need.",
            "The Key is the label on each book.",
            "The Value contains the actual requested content."
        ]
        self.setup_layout("The Mechanism: Queries, Keys, and Values", lecture_lines)
        
        # Initialize mobjects using assets
        q_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/book.svg")
        k_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/book.svg")
        v_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/book.svg")
        
        q_label = Text("Query (Q)", font_size=20)
        k_label = Text("Key (K)", font_size=20)
        v_label = Text("Value (V)", font_size=20)
        
        # Group
        q_group = VGroup(q_icon, q_label).arrange(DOWN)
        k_group = VGroup(k_icon, k_label).arrange(DOWN)
        v_group = VGroup(v_icon, v_label).arrange(DOWN)
        
        # Positioning as requested per feedback (Issue 29)
        self.place_at_grid(q_group, 'B2', scale_factor=0.7)
        self.place_at_grid(k_group, 'C2', scale_factor=0.7)
        self.place_at_grid(v_group, 'D2', scale_factor=0.7)
        
        self.play(FadeIn(q_group), FadeIn(k_group), FadeIn(v_group))

        # === Animation for Lecture Line 1 ===
        # Show three labeled boxes: Query (Q), Key (K), Value (V), each represented by book.svg. Change color to #32CD32 (LimeGreen).
        self.play(self.lecture[0].animate.set_color("#32CD32")) 
        self.play(q_icon.animate.set_color("#32CD32"), k_icon.animate.set_color("#32CD32"), v_icon.animate.set_color("#32CD32"))
        
        # === Animation for Lecture Line 2 ===
        # Animate Q moving towards K for comparison. Change color to #FFFFFF (White).
        # Highlight matching intensity between Q and K. Change color to #00CED1 (DarkTurquoise).
        self.play(self.lecture[1].animate.set_color("#FFFFFF")) 
        self.play(q_group.animate.next_to(k_group, RIGHT, buff=0.5))
        self.play(k_icon.animate.set_color("#FFFFFF"))
        self.play(q_icon.animate.set_color("#00CED1"), k_icon.animate.set_color("#00CED1"))
        
        # === Animation for Lecture Line 3 ===
        # Morph V based on the match result of Q and K. Change color to #FFD700 (Gold).
        # Show weighted combination of values forming output, derived from the book.svg. Change color to #FF4500 (OrangeRed).
        self.play(self.lecture[2].animate.set_color("#FFD700")) 
        self.play(v_icon.animate.set_color("#FFD700"))
        self.play(v_group.animate.scale(1.2))
        self.play(v_icon.animate.set_color("#FF4500"))
        
        self.wait(2)
