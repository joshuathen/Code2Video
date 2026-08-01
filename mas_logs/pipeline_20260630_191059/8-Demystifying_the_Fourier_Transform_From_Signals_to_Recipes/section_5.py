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
        # Data from shared state
        title = "The Center of Mass (The Peak)"
        lecture_lines = [
            "Think of the wrapped signal as a physical wire.",
            "We calculate its center of mass as we spin.",
            "Usually, the center of mass stays near the origin.",
            "When we hit a hidden frequency, it jumps outward.",
            "This sudden \"peak\" signals we found an ingredient."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Colors
        COLOR_WIRE = "#ECF0F1"  # White wire
        COLOR_DOT = "#F1C40F"   # Yellow center of mass dot
        
        # Asset path
        DOT_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/dot.svg"
        
        # === Animation for Lecture Line 1 ===
        # "Think of the wrapped signal as a physical wire."
        self.lecture[0].set_color(YELLOW)
        
        # Create a complex winding wire (balanced state)
        # Using a parametric function to simulate a wrapped signal
        def get_wrapped_wire(offset_vec=np.array([0,0,0]), freq_mod=1.0):
            return ParametricFunction(
                lambda t: np.array([
                    (1.5 + 0.3 * np.sin(5 * t * freq_mod)) * np.cos(t),
                    (1.5 + 0.3 * np.sin(5 * t * freq_mod)) * np.sin(t),
                    0
                ]) + offset_vec,
                t_range=[0, 2 * PI],
                color=COLOR_WIRE
            )
            
        wire = get_wrapped_wire()
        # Fix Issue 54: Position in area B2-E5
        self.place_in_area(wire, 'B2', 'E5', scale_factor=1.0)
        wire_center = wire.get_center()
        
        self.play(Create(wire))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We calculate its center of mass as we spin."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Animation 1: Place the yellow dot (SVG asset)
        # Fix Issue 45 & 55: Use SVG asset and position at D4
        dot = SVGMobject(DOT_ASSET)
        dot.set_color(COLOR_DOT)
        self.place_at_grid(dot, 'D4', scale_factor=0.8)
        
        self.play(FadeIn(dot))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Usually, the center of mass stays near the origin."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Animation 2: Animate the dot vibrating near the center (D4)
        jitter_tracker = ValueTracker(0)
        origin_pos = self.grid['D4'].copy()
        
        # Add updater to dot for jitter
        dot.add_updater(lambda d: d.move_to(
            origin_pos + 
            np.array([0.15 * np.sin(jitter_tracker.get_value() * 12), 
                      0.15 * np.cos(jitter_tracker.get_value() * 9), 
                      0])
        ))
        
        self.play(jitter_tracker.animate.set_value(2), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "When we hit a hidden frequency, it jumps outward."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Prepare the "unbalanced" wire
        unbalanced_wire = ParametricFunction(
            lambda t: np.array([
                (1.5 + 1.2 * np.cos(t)) * np.cos(t), # Lopsided towards positive x
                (1.5 + 1.2 * np.cos(t)) * np.sin(t),
                0
            ]) + wire_center, # Keep relative to the same area center
            t_range=[0, 2 * PI],
            color=COLOR_WIRE
        )
        # Note: unbalanced_wire is already shifted by wire_center, 
        # but place_in_area might be more consistent. 
        # However, we want the wire to stay centered in its area while its geometry changes.
        
        # Stop the jitter updater before the jump
        dot.clear_updaters()
        
        # Fix Issue 56: Jump target D6
        jump_target = self.grid['D6']
        
        self.play(
            Transform(wire, unbalanced_wire),
            dot.animate.move_to(jump_target),
            run_time=1.5,
            rate_func=slow_into
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "This sudden 'peak' signals we found an ingredient."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Pulse the dot to emphasize the "peak"
        self.play(
            dot.animate.scale(1.5).set_color(WHITE),
            Flash(jump_target, color=COLOR_DOT, line_length=0.3, num_lines=8),
            run_time=0.5
        )
        self.play(
            dot.animate.scale(1/1.5).set_color(COLOR_DOT),
            run_time=0.5
        )
        
        self.wait(2)

        # Reset colors for final state
        self.lecture[4].set_color(WHITE)
        self.wait(1)
