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
        self.setup_layout("The Analogy: How We Learn from Mistakes", [
            "Learning is adjusting settings to reduce error.", 
            "Loss functions provide a score for our error.", 
            "Think of an archer adjusting their aim."
        ])
        
        # Assets
        archer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/archer.svg", color=WHITE)
        bow = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bow.svg", color=WHITE)
        target = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/target.svg", color=WHITE)
        arrow = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/arrow.svg", color=WHITE)
        
        # Fix 26: Target and bullseye
        self.place_at_grid(target, 'C2', scale_factor=0.9)
        
        # Fix 28: Loss label
        loss_label = Text("Loss", color=RED).scale(0.5)
        self.place_at_grid(loss_label, 'D2', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        # Using assets: archer, bow, target
        archer_group = VGroup(archer, bow).arrange(RIGHT).scale(0.5)
        self.place_at_grid(archer_group, "C5")
        self.play(FadeIn(archer_group), FadeIn(target))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF4500"))
        
        # Fix 27: Arrow positioning
        self.place_at_grid(arrow, 'C2', scale_factor=0.8)
        arrow_high = arrow.copy().move_to(self.grid["B2"])
        
        loss_line = DashedLine(start=target.get_center(), end=arrow_high.get_center(), color=RED)
        self.play(Create(arrow_high))
        self.play(Create(loss_line), FadeIn(loss_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00BFFF"))
        # Adjusting aim (archer moves slightly)
        self.play(archer_group.animate.shift(DOWN*0.5))
        self.play(FadeOut(arrow_high), FadeOut(loss_line), FadeOut(loss_label))
        
        # Repeat shot
        arrow_fixed = arrow.copy().move_to(self.grid["C2"])
        self.play(FadeIn(arrow_fixed), run_time=1.5)
        self.play(arrow_fixed.animate.set_color("#32CD32"))
