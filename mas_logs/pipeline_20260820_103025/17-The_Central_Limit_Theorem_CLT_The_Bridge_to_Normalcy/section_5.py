from manim import *
import numpy as np

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
        self.setup_layout("The 'Why It Matters' Application", ["CLT lets us infer population properties.", "We predict vast outcomes from small samples.", "This saves time, effort, and resources."])
        
        # === Animation for Lecture Line 1 ===
        # Show factory assets [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/factory.svg] 
        # and assembly assets [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/assembly.svg]
        factory = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/factory.svg", color="#FF5733")
        assembly = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/assembly.svg", color="#FF5733")
        factory_assembly = VGroup(factory, assembly).arrange(RIGHT, buff=0.2)
        
        # Apply fix from issue 33: use C2-E5 to avoid text overlap
        self.place_in_area(factory_assembly, "C2", "E5", scale_factor=0.6)
        self.play(FadeIn(factory_assembly))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        # Overlay a normal curve [Asset references used]
        axes = Axes(x_range=[-3, 3, 1], y_range=[0, 1, 0.5], axis_config={"include_tip": False})
        bell_curve = axes.plot(lambda x: np.exp(-x**2/2), color="#4287f5")
        
        # Apply fix from issue 34: use A3-F6
        self.place_in_area(VGroup(axes, bell_curve), "A3", "F6", scale_factor=0.5)
        self.play(Create(axes), Create(bell_curve))
        self.lecture[1].set_color("#4287f5")

        # === Animation for Lecture Line 3 ===
        # Highlight SD area
        sd_area = axes.get_area(bell_curve, x_range=[-1, 1], color="#FFFF00", opacity=0.4)
        
        # Apply fix from issue 35: add summary label at A5
        summary_label = Text("Quality Control", color="#FFFF00", font_size=24)
        self.place_at_grid(summary_label, "A5", scale_factor=0.7)
        
        self.play(FadeIn(sd_area), Write(summary_label))
        self.lecture[2].set_color("#FFFF00")
        
        self.wait(2)
