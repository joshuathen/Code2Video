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
        self.setup_layout("Synthesis and Summary", [
            "Attention handles context; MLPs store facts.", 
            "Facts are distributed across weight patterns.", 
            "Together, they form an intelligent system."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/brain.svg]
        brain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/brain.svg", color="#3498DB")
        self.place_in_area(brain, 'A3', 'C5', scale_factor=0.9)
        
        self.play(FadeIn(brain))
        self.lecture[0].set_color("#3498DB")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Distributed patterns across weights
        pattern_rects = VGroup(*[Rectangle(width=0.4, height=0.2, color="#E74C3C", fill_opacity=0.8) for _ in range(12)])
        pattern_rects.arrange_in_grid(4, 3, buff=0.05)
        self.place_in_area(pattern_rects, 'D3', 'F5', scale_factor=0.9)
        
        label = Text("Weight Patterns", font_size=18, color=WHITE).scale(0.7)
        label.next_to(pattern_rects, RIGHT) # Tethered via next_to
        
        self.play(FadeIn(pattern_rects), Write(label))
        self.lecture[1].set_color("#E74C3C")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final System Synthesis using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/filing.svg]
        filing = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/filing.svg", color="#2ECC71")
        self.place_at_grid(filing, 'B5', scale_factor=0.75)
        
        self.play(GrowFromCenter(filing))
        self.lecture[2].set_color("#2ECC71")
        self.wait(2)
