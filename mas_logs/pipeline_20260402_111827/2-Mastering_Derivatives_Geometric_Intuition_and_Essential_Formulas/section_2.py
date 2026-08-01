from manim import *
import os

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
        # Initialize the layout with title and lecture lines
        lecture_lines = [
            "Look at this point on a curved rollercoaster track.",
            "A tangent line touches the curve at this spot.",
            "Let's zoom in closely on this specific point.",
            "Up close, the curve looks like a straight line.",
            "This local linearity defines the derivative's value here."
        ]
        self.setup_layout("The Geometric Intuition: Local Linearity", lecture_lines)
        
        # Ensure UI elements stay on top of the zooming graph
        self.lecture.set_z_index(10)
        self.title.set_z_index(10)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Issue 32: Occupy area A3-E6
        axes = Axes(
            x_range=[-0.2, 2.2, 1],
            y_range=[-0.2, 4.2, 1],
            x_length=5,
            y_length=5,
            axis_config={"color": BLUE_E, "include_tip": False}
        )
        self.place_in_area(axes, 'A3', 'E6', scale_factor=1.0)
        
        parabola = axes.plot(lambda x: x**2, x_range=[0, 2.1], color=WHITE)
        # Point P at (1,1)
        p_coords = axes.c2p(1, 1)
        dot = Dot(p_coords, color=WHITE, radius=0.08)
        
        # Issue 26: Integrate rollercoaster icon asset
        asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/rollercoaster.svg"
        if os.path.exists(asset_path):
            rollercoaster = SVGMobject(asset_path)
        else:
            # Fallback if asset is missing (though MAS expects it to be there)
            rollercoaster = VMobject() 
            
        rollercoaster.scale(0.3)
        rollercoaster.move_to(p_coords)
        rollercoaster.set_color(WHITE)
        
        self.play(Create(axes), Create(parabola), run_time=1.5)
        self.play(FadeIn(dot), FadeIn(rollercoaster))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFD700") # Golden color for tangent line
        
        # Tangent line at x=1: y = 2x - 1
        tangent = axes.plot(lambda x: 2*x - 1, x_range=[0.4, 1.6], color="#FFD700")
        self.play(Create(tangent))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Group components for a unified zoom effect
        zoom_group = VGroup(axes, parabola, tangent, dot, rollercoaster)
        zoom_center = dot.get_center()
        
        # Execute the zoom by a factor of 20. 
        # Adjust stroke widths and component scales to keep them visible but not massive.
        self.play(
            zoom_group.animate.scale(20, about_point=zoom_center),
            parabola.animate.set_stroke_width(2),
            tangent.animate.set_stroke_width(2),
            axes.animate.set_stroke_width(1),
            dot.animate.scale(0.05), # 1/20 to keep visual size constant
            rollercoaster.animate.scale(0.05),
            run_time=4
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Visual focus on the line segment
        self.play(Indicate(tangent, color="#FFD700", scale_factor=1.02))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        label = Text("Local Linearity", font_size=24, color=WHITE)
        # Issue 31: Place in area F4-F6
        self.place_in_area(label, 'F4', 'F6', scale_factor=0.8)
        label.set_z_index(10)
        
        self.play(Write(label))
        self.wait(3)

        # Final state cleanup
        self.lecture[4].set_color(WHITE)
