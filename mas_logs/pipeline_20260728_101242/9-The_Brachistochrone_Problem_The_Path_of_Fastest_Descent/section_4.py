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

class Section4Scene(TeachingScene):
    def construct(self):
        title = "The Light Analogy Solution"
        lines = [
            "Imagine gravity as layers of increasing light speed.",
            "Each layer bends the path slightly to save time.",
            "Snell’s Law dictates the exact ratio of the curve.",
            "The math reveals a specific differential equation.",
            "This equation defines our optimal, fastest path."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create thin horizontal layers with increasing brightness
        num_layers = 12
        layers = VGroup()
        
        # Area for layers: covers grid area from A1 to F6
        width = 5.5
        total_height = 5.5
        layer_height = total_height / num_layers
        
        for i in range(num_layers):
            # Calculate color brightness: increasing with depth (downward)
            alpha = 0.1 + (i / num_layers) * 0.4
            layer_color = interpolate_color(BLUE_E, WHITE, alpha)
            layer = Rectangle(
                width=width, 
                height=layer_height, 
                fill_color=layer_color, 
                fill_opacity=0.2, 
                stroke_width=0.5,
                stroke_color=GRAY
            )
            # Position layers starting from top of grid
            # Center of grid area is roughly x=3.0, y=-0.3
            y_pos = 2.5 - (i * layer_height) - (layer_height / 2)
            layer.move_to([3.0, y_pos, 0])
            layers.add(layer)
            
        self.play(FadeIn(layers, lag_ratio=0.1), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Trace a light ray bending through each layer
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/light.svg]
        light_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/light.svg", color="#FFFF00").scale(0.2)
        
        ray_points = []
        # Start at top left-ish of the grid area
        start_point = self.grid["A1"] + UP * 0.2
        ray_points.append(start_point)
        
        current_p = start_point
        # Simple refraction model: angle increases relative to vertical as speed increases
        for i in range(num_layers):
            angle = np.deg2rad(-20 - (i * 5.5)) 
            # Step size to cross one layer vertically
            step_mag = (layer_height) / np.abs(np.sin(angle))
            step = np.array([np.cos(angle), np.sin(angle), 0]) * step_mag
            current_p = current_p + step
            ray_points.append(current_p)
            
        ray = VMobject(color="#FFFF00", stroke_width=4)
        ray.set_points_as_corners(ray_points)
        
        self.add(light_icon)
        self.play(
            Create(ray), 
            MoveAlongPath(light_icon, ray), 
            run_time=3, 
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Smooth the jagged ray into a continuous curve
        smooth_ray = VMobject(color="#FFFF00", stroke_width=4)
        smooth_ray.set_points_smoothly(ray_points)
        
        # [Issue 27] Snell's Law at B2, scale 0.9
        snells_law = MathTex(r"\frac{\sin\theta}{v} = k", font_size=36)
        self.place_at_grid(snells_law, "B2", scale_factor=0.9)
        
        self.play(
            Transform(ray, smooth_ray),
            Write(snells_law)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # [Issue 27] v_sub at C2, scale 0.9
        v_sub = MathTex(r"v = \sqrt{2gy}", font_size=28)
        self.place_at_grid(v_sub, "C2", scale_factor=0.9)
        
        # [Issue 28] final_eq at B2, scale 1.1
        final_formula_text = r"\frac{\sin\theta}{\sqrt{y}} = k"
        final_eq = MathTex(final_formula_text, font_size=36)
        self.place_at_grid(final_eq, "B2", scale_factor=1.1)
        
        self.play(FadeIn(v_sub, shift=DOWN))
        self.wait(1)
        self.play(
            Transform(snells_law, final_eq),
            FadeOut(v_sub)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # [Issue 29] highlight_eq in area B1-B3, scale 1.0
        highlight_eq = MathTex(final_formula_text, color="#00FFFF", font_size=40)
        self.place_in_area(highlight_eq, 'B1', 'B3', scale_factor=1.0)
        
        rect = SurroundingRectangle(highlight_eq, color="#00FFFF", buff=0.2)
        
        self.play(
            Transform(snells_law, highlight_eq),
            Create(rect)
        )
        self.wait(2)

        # Transition out
        self.play(
            *[FadeOut(m) for m in [layers, ray, light_icon, snells_law, rect]]
        )
        self.wait(1)
