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
        lecture_lines = [
            "MLP layers contain two linear projections.",
            "A non-linear activation sits between them.",
            "The first layer identifies input patterns.",
            "The second layer writes output values.",
            "This acts like a memory slot."
        ]
        self.setup_layout("The MLP as a Key-Value Memory Bank", lecture_lines)
        
        # Color palette
        c1, c2, c3 = "#FFD700", "#32CD32", "#FF4500"
        
        # Elements
        mem_bank = Rectangle(width=3, height=3, color=WHITE)
        self.place_in_area(mem_bank, 'B3', 'E5', scale_factor=1.0)
        
        # Asset
        slot_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slot.svg", color=WHITE)
        self.place_at_grid(slot_icon, 'C4', scale_factor=0.5)
        
        key_vec = Arrow(start=LEFT, end=RIGHT, color=c1).scale(0.5)
        val_vec = Arrow(start=LEFT, end=RIGHT, color=c2).scale(0.5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.play(FadeIn(mem_bank), FadeIn(slot_icon))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(c3)
        mid_line = Line(start=LEFT, end=RIGHT, color=c3).scale(0.5)
        self.place_at_grid(mid_line, 'C4')
        self.play(Create(mid_line))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(c1)
        self.place_at_grid(key_vec, 'B2', scale_factor=0.9)
        self.play(Create(key_vec))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(c2)
        self.place_at_grid(val_vec, 'D6', scale_factor=0.9)
        self.play(Create(val_vec))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        slot_label = Text("Memory Slot", font_size=20, color=WHITE)
        self.place_at_grid(slot_label, 'E4', scale_factor=1.0)
        self.play(Write(slot_label))
        self.wait(2)
